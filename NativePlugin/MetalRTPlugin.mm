// Metal hardware ray tracing path tracer plugin for Unity (stage 1: URP Lit
// materials only)
//
// Builds per-mesh primitive acceleration structures (BLAS) directly from
// Unity mesh GPU buffers, combines them into an instance acceleration
// structure (TLAS), and path-traces the scene with a wavefront-style
// per-bounce kernel pipeline (RayGen -> [Intersect -> Shade]*N -> Resolve).
// Surface evaluation is isolated in the Shade kernel behind a SurfaceData
// boundary so that stage 2 can replace it with Unity-compiled material
// evaluation. Mesh vertex/index buffers and material base maps are accessed
// bindlessly (Metal 3 gpuAddress / gpuResourceID).
//
// Execution paths:
// - Synchronous (private queue + waitUntilCompleted): one-time BLAS builds,
//   probe tests, material table setup.
// - Render thread: the per-frame TLAS rebuild and the path tracing pipeline
//   are encoded into Unity's own Metal command buffer from a plugin render
//   event (CommandBuffer.IssuePluginEventAndData).

#import <Metal/Metal.h>

#include "IUnityInterface.h"
#include "IUnityGraphics.h"
#include "IUnityGraphicsMetal.h"

#include <atomic>
#include <cstring>
#include <string>
#include <vector>

namespace {

IUnityGraphicsMetalV1* s_Metal;

id<MTLDevice> s_Device;
id<MTLCommandQueue> s_Queue;

id<MTLComputePipelineState> s_ClearPipeline;
id<MTLComputePipelineState> s_RayGenPipeline;
id<MTLComputePipelineState> s_IntersectPipeline;
id<MTLComputePipelineState> s_ShadePipeline;
id<MTLComputePipelineState> s_ResolvePipeline;
id<MTLComputePipelineState> s_ProbePipeline;

struct MeshEntry
{
    id<MTLAccelerationStructure> blas;
    id<MTLBuffer> vertexBuffer;
    id<MTLBuffer> indexBuffer;
    uint32_t vertexStride, positionOffset, normalOffset, uvOffset;
    uint32_t indexFormat;
};

// Written on the main thread during setup (before any render event fires),
// read from the render thread afterwards.
std::vector<MeshEntry> s_Meshes;
id<MTLBuffer> s_MaterialBuffer;
std::vector<id<MTLTexture>> s_MaterialTextures;
id<MTLAccelerationStructure> s_InstanceAS; // synchronous path only
id<MTLBuffer> s_InstanceInfo;              // synchronous path only

// Per-frame path tracing resources (render thread only)
id<MTLBuffer> s_PathBuffer;
id<MTLBuffer> s_HitBuffer;
id<MTLBuffer> s_AccumBuffer;
uint32_t s_BufferWidth, s_BufferHeight;

std::atomic<int> s_EventFrames {0};

std::string s_LastError;

constexpr uint32_t kNoAttribute = 0xffffffffu;
constexpr int kMaxEventInstances = 16;
constexpr uint32_t kMaxBounceLimit = 8;

// Must match TraceParams in MetalRTPlugin.cs and in the shader source.
struct TraceParams
{
    float originTan[4];   // xyz: ray origin (world space), w: tan(fovY/2)
    float rightAspect[4]; // xyz: camera right (world space), w: aspect
    float up[4];          // xyz: camera up (world space)
    float forward[4];     // xyz: camera forward (world space)
    uint32_t width, height, pad0, pad1;
};

// Must match ProbeResult in MetalRTPlugin.cs.
struct ProbeResult
{
    float hit, distance, primitiveIndex, instanceIndex;
    float barycentric[2], pad[2];
};

// Must match InstanceDesc in MetalRTPlugin.cs.
struct InstanceDesc
{
    int32_t meshIndex;
    int32_t materialIndex;
    float pad[2];
    float objectToWorld[3][4]; // rows of the 3x4 object-to-world matrix
    float normalMatrix[3][4];  // rows (xyz used) of the world normal matrix
};

// Must match MaterialDesc in MetalRTPlugin.cs.
struct MaterialDesc
{
    float baseColor[4]; // linear
    float emission[4];  // linear
    float baseMapST[4]; // xy: scale, zw: offset
    float metallic, smoothness;
    uint32_t hasBaseMap, pad;
    uint64_t texture;   // MTLTexture pointer or 0
    uint64_t pad2;
};

// Debug flags in EventData (must match PathTracerTest.cs).
enum DebugFlags
{
    kDebugDiffuseOnly = 1, // disable the specular lobe (analytic tests)
};

// Must match the event data blob layout in MetalRTPlugin.cs.
struct EventData
{
    TraceParams params;    // 80
    uint64_t texture;      // 80: RenderTexture native pointer
    int32_t instanceCount; // 88
    uint32_t frameIndex;   // 92
    float envColor[4];     // 96: linear radiance of the uniform environment
    float lightDir[4];     // 112: xyz: direction the light travels (normalized)
    float lightColor[4];   // 128: linear color * intensity
    uint32_t reset;        // 144
    uint32_t maxBounces;   // 148
    uint32_t linearOutput; // 152: 1 = skip tonemap/sRGB (analytic tests)
    uint32_t debugFlags;   // 156
    float exposure;        // 160
    float pad[3];          // 164..176
    InstanceDesc instances[kMaxEventInstances];
};

// Shader-side per-instance record; must match InstanceInfo in the shader.
struct GpuInstanceInfo
{
    uint64_t vertices, indices;
    uint32_t vertexStride, positionOffset, normalOffset, uvOffset;
    uint32_t indexFormat, materialIndex, pad0, pad1;
    float normalMatrix[3][4];
};

// Shader-side material record; must match GpuMaterial in the shader.
// The base map is referenced by MTLResourceID (Metal 3 bindless).
struct GpuMaterial
{
    float baseColor[4];
    float emission[4];
    float baseMapST[4];
    float metallic, smoothness;
    uint32_t hasBaseMap, pad;
    MTLResourceID baseMap;
    uint64_t pad2;
};

// Shader-side per-frame constants; must match FrameConstants in the shader.
struct FrameConstants
{
    TraceParams cam;
    float envColor[4];
    float lightDir[4];
    float lightColor[4];
    uint32_t frameIndex, maxBounces, linearOutput, debugFlags;
    float exposure;
    uint32_t pad[3];
};

constexpr const char* kShaderSource = R"msl(
#include <metal_stdlib>
#include <metal_raytracing>

using namespace metal;
using namespace metal::raytracing;

constant float kPi = 3.14159265358979;
constant float kRayEpsilon = 1e-3;

struct TraceParams
{
    float4 originTan;
    float4 rightAspect;
    float4 up;
    float4 forward;
    uint width, height, pad0, pad1;
};

struct FrameConstants
{
    TraceParams cam;
    float4 envColor;
    float4 lightDir;
    float4 lightColor;
    uint frameIndex, maxBounces, linearOutput, debugFlags;
    float exposure;
    uint pad0, pad1, pad2;
};

struct ProbeResult
{
    float hit, distance, primitiveIndex, instanceIndex;
    float2 barycentric, pad;
};

struct InstanceInfo
{
    device const uchar* vertices;
    device const uchar* indices;
    uint vertexStride, positionOffset, normalOffset, uvOffset;
    uint indexFormat, materialIndex, pad0, pad1;
    float4 normalMatrix0, normalMatrix1, normalMatrix2;
};

struct GpuMaterial
{
    float4 baseColor;
    float4 emission;
    float4 baseMapST;
    float metallic, smoothness;
    uint hasBaseMap, pad;
    texture2d<float> baseMap;
    ulong pad2;
};

// Per-pixel path state (w components carry rng state / alive flag).
struct PathState
{
    float4 origin;     // w: unused
    float4 direction;  // w: unused
    float4 throughput; // w: rng state (as_type)
    float4 radiance;   // w: alive flag
};

struct HitRecord
{
    float distance;
    uint instanceIndex, primitiveIndex, hit;
    float2 barycentric, pad;
};

// --- RNG (PCG) ---

static uint PcgNext(thread uint& state)
{
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28) + 4u)) ^ state) * 277803737u;
    return (word >> 22) ^ word;
}

static float Rand01(thread uint& state)
  { return PcgNext(state) / 4294967296.0; }

static uint SeedRng(uint2 pixel, uint frame)
{
    uint s = pixel.x * 1973u + pixel.y * 9277u + frame * 26699u + 1u;
    PcgNext(s);
    return s;
}

// --- Geometry fetch ---

static uint3 LoadTriangleIndices(constant InstanceInfo& info, uint prim)
{
    if (info.indexFormat == 0)
    {
        device const ushort* p = (device const ushort*)info.indices;
        return uint3(p[prim * 3], p[prim * 3 + 1], p[prim * 3 + 2]);
    }
    else
    {
        device const uint* p = (device const uint*)info.indices;
        return uint3(p[prim * 3], p[prim * 3 + 1], p[prim * 3 + 2]);
    }
}

static float3 LoadFloat3Attr
  (constant InstanceInfo& info, uint offset, uint index)
{
    device const packed_float3* p = (device const packed_float3*)
      (info.vertices + info.vertexStride * index + offset);
    return *p;
}

static float2 LoadFloat2Attr
  (constant InstanceInfo& info, uint offset, uint index)
{
    device const packed_float2* p = (device const packed_float2*)
      (info.vertices + info.vertexStride * index + offset);
    return *p;
}

// --- Surface evaluation (stage 2 boundary: this is the part that will be
// --- replaced by Unity-compiled material evaluation) ---

struct SurfaceData
{
    float3 position;   // world space
    float3 normal;     // world space shading normal (faces the ray origin)
    float3 baseColor;  // linear, texture applied
    float3 emission;   // linear
    float metallic;
    float smoothness;
};

static SurfaceData EvaluateSurface
  (constant InstanceInfo& info, constant GpuMaterial* materials,
   thread const HitRecord& hit, float3 rayOrigin, float3 rayDir)
{
    uint3 tri = LoadTriangleIndices(info, hit.primitiveIndex);
    float3 bary = float3(1 - hit.barycentric.x - hit.barycentric.y,
                         hit.barycentric.x, hit.barycentric.y);

    // Geometric normal (object space)
    float3 p0 = LoadFloat3Attr(info, info.positionOffset, tri.x);
    float3 p1 = LoadFloat3Attr(info, info.positionOffset, tri.y);
    float3 p2 = LoadFloat3Attr(info, info.positionOffset, tri.z);
    float3 n = cross(p1 - p0, p2 - p0);

    // Interpolated vertex normal when available
    if (info.normalOffset != 0xffffffffu)
    {
        float3 n0 = LoadFloat3Attr(info, info.normalOffset, tri.x);
        float3 n1 = LoadFloat3Attr(info, info.normalOffset, tri.y);
        float3 n2 = LoadFloat3Attr(info, info.normalOffset, tri.z);
        float3 sn = n0 * bary.x + n1 * bary.y + n2 * bary.z;
        if (dot(sn, n) < 0) n = -n;
        n = sn;
    }

    // Object -> world normal transform (inverse transpose rows)
    n = float3(dot(info.normalMatrix0.xyz, n),
               dot(info.normalMatrix1.xyz, n),
               dot(info.normalMatrix2.xyz, n));
    n = normalize(n);
    if (dot(n, rayDir) > 0) n = -n;

    constant GpuMaterial& mat = materials[info.materialIndex];

    float3 baseColor = mat.baseColor.xyz;
    if (mat.hasBaseMap != 0 && info.uvOffset != 0xffffffffu)
    {
        float2 uv0 = LoadFloat2Attr(info, info.uvOffset, tri.x);
        float2 uv1 = LoadFloat2Attr(info, info.uvOffset, tri.y);
        float2 uv2 = LoadFloat2Attr(info, info.uvOffset, tri.z);
        float2 uv = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;
        uv = uv * mat.baseMapST.xy + mat.baseMapST.zw;
        constexpr sampler smp(address::repeat, filter::linear, mip_filter::linear);
        baseColor *= mat.baseMap.sample(smp, uv, level(2)).xyz;
    }

    SurfaceData surf;
    surf.position = rayOrigin + rayDir * hit.distance;
    surf.normal = n;
    surf.baseColor = baseColor;
    surf.emission = mat.emission.xyz;
    surf.metallic = mat.metallic;
    surf.smoothness = mat.smoothness;
    return surf;
}

// --- BSDF: URP Lit compatible metallic-roughness GGX + Lambert ---

struct Bsdf
{
    float3 diffuse; // Lambert albedo
    float3 f0;      // specular reflectance at normal incidence
    float alpha;    // GGX alpha (perceptual roughness squared)
};

static Bsdf MakeBsdf(thread const SurfaceData& surf, uint debugFlags)
{
    Bsdf b;
    b.diffuse = surf.baseColor * (1 - surf.metallic);
    b.f0 = mix(float3(0.04), surf.baseColor, surf.metallic);
    float perceptual = 1 - surf.smoothness;
    b.alpha = max(perceptual * perceptual, 1e-3);
    if (debugFlags & 1) // diffuse only (analytic tests)
        b.f0 = float3(0);
    return b;
}

static float GgxD(float alpha, float noh)
{
    float a2 = alpha * alpha;
    float d = noh * noh * (a2 - 1) + 1;
    return a2 / max(kPi * d * d, 1e-7);
}

static float SmithG(float alpha, float nov, float nol)
{
    float a2 = alpha * alpha;
    float gv = nol * sqrt(nov * nov * (1 - a2) + a2);
    float gl = nov * sqrt(nol * nol * (1 - a2) + a2);
    return 2 * nov * nol / max(gv + gl, 1e-7);
}

static float3 Fresnel(float3 f0, float voh)
  { return f0 + (1 - f0) * pow(1 - voh, 5.0); }

// Full BSDF value for given directions (used by next event estimation).
static float3 EvalBsdf(thread const Bsdf& b, float3 n, float3 wo, float3 wi)
{
    float nol = dot(n, wi);
    float nov = dot(n, wo);
    if (nol <= 0 || nov <= 0) return float3(0);
    float3 h = normalize(wo + wi);
    float3 spec = GgxD(b.alpha, dot(n, h)) * SmithG(b.alpha, nov, nol) *
                  Fresnel(b.f0, dot(wo, h)) / (4 * nov * nol);
    return b.diffuse / kPi + spec;
}

static float3x3 MakeBasis(float3 n)
{
    float3 t = abs(n.y) < 0.99 ? normalize(cross(n, float3(0, 1, 0)))
                               : float3(1, 0, 0);
    float3 bt = cross(t, n);
    return float3x3(t, bt, n); // columns: tangent, bitangent, normal
}

static float3 SampleCosine(thread uint& rng, float3 n)
{
    float u1 = Rand01(rng), u2 = Rand01(rng);
    float r = sqrt(u1), phi = 2 * kPi * u2;
    float3 local = float3(r * cos(phi), r * sin(phi),
                          sqrt(max(0.0, 1 - u1)));
    return MakeBasis(n) * local;
}

static float3 SampleGgxHalf(thread uint& rng, float3 n, float alpha)
{
    float u1 = Rand01(rng), u2 = Rand01(rng);
    float phi = 2 * kPi * u1;
    float ct = sqrt((1 - u2) / (1 + (alpha * alpha - 1) * u2));
    float st = sqrt(max(0.0, 1 - ct * ct));
    float3 local = float3(st * cos(phi), st * sin(phi), ct);
    return MakeBasis(n) * local;
}

static float Luminance(float3 c)
  { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

// Samples the next bounce direction and returns the throughput multiplier
// (f * cos / pdf, including the lobe selection probability).
static float3 SampleBsdf(thread const Bsdf& b, thread uint& rng,
                         float3 n, float3 wo, thread float3& wi)
{
    float wDiff = Luminance(b.diffuse);
    float wSpec = Luminance(b.f0);
    float pSpec = wDiff + wSpec > 0 ?
      clamp(wSpec / (wDiff + wSpec), 0.0, 1.0) : 0.0;

    if (Rand01(rng) < pSpec)
    {
        float3 h = SampleGgxHalf(rng, n, b.alpha);
        wi = reflect(-wo, h);
        float nol = dot(n, wi);
        float nov = dot(n, wo);
        if (nol <= 0 || nov <= 0) return float3(0);
        // pdf = D * NoH / (4 * VoH); f = D*G*F / (4*NoV*NoL)
        // f * cos / pdf = G * F * VoH / (NoV * NoH)
        float voh = max(dot(wo, h), 1e-4);
        float noh = max(dot(n, h), 1e-4);
        float3 w = SmithG(b.alpha, nov, nol) * Fresnel(b.f0, voh) *
                   voh / (nov * noh);
        return w / max(pSpec, 1e-3);
    }
    else
    {
        wi = SampleCosine(rng, n);
        if (dot(n, wi) <= 0) return float3(0);
        // pdf = cos/pi; f = diffuse/pi -> f * cos / pdf = diffuse
        return b.diffuse / max(1 - pSpec, 1e-3);
    }
}

// --- Kernels ---

kernel void ClearAccum
  (device float4* accum [[buffer(0)]],
   constant FrameConstants& frame [[buffer(1)]],
   uint2 id [[thread_position_in_grid]])
{
    if (id.x >= frame.cam.width || id.y >= frame.cam.height) return;
    accum[id.y * frame.cam.width + id.x] = float4(0);
}

kernel void RayGen
  (device PathState* paths [[buffer(0)]],
   constant FrameConstants& frame [[buffer(1)]],
   uint2 id [[thread_position_in_grid]])
{
    if (id.x >= frame.cam.width || id.y >= frame.cam.height) return;

    uint rng = SeedRng(id, frame.frameIndex);
    float2 jitter = float2(Rand01(rng), Rand01(rng));
    float2 ndc = (float2(id) + jitter) /
                 float2(frame.cam.width, frame.cam.height) * 2 - 1;

    float tanFov = frame.cam.originTan.w;
    float aspect = frame.cam.rightAspect.w;
    float3 dir = frame.cam.forward.xyz +
                 frame.cam.rightAspect.xyz * (ndc.x * tanFov * aspect) +
                 frame.cam.up.xyz * (ndc.y * tanFov);

    PathState path;
    path.origin = float4(frame.cam.originTan.xyz, 0);
    path.direction = float4(normalize(dir), 0);
    path.throughput = float4(1, 1, 1, as_type<float>(rng));
    path.radiance = float4(0, 0, 0, 1);
    paths[id.y * frame.cam.width + id.x] = path;
}

kernel void Intersect
  (instance_acceleration_structure accel [[buffer(0)]],
   device PathState* paths [[buffer(1)]],
   device HitRecord* hits [[buffer(2)]],
   constant FrameConstants& frame [[buffer(3)]],
   uint2 id [[thread_position_in_grid]])
{
    if (id.x >= frame.cam.width || id.y >= frame.cam.height) return;
    uint idx = id.y * frame.cam.width + id.x;

    PathState path = paths[idx];
    if (path.radiance.w == 0) return;

    ray r;
    r.origin = path.origin.xyz;
    r.direction = path.direction.xyz;
    r.min_distance = 0;
    r.max_distance = INFINITY;

    intersector<triangle_data, instancing> isect;
    intersection_result<triangle_data, instancing> hit =
      isect.intersect(r, accel, 0xffu);

    HitRecord rec = {};
    if (hit.type == intersection_type::triangle)
    {
        rec.hit = 1;
        rec.distance = hit.distance;
        rec.instanceIndex = hit.instance_id;
        rec.primitiveIndex = hit.primitive_id;
        rec.barycentric = hit.triangle_barycentric_coord;
    }
    hits[idx] = rec;
}

kernel void Shade
  (instance_acceleration_structure accel [[buffer(0)]],
   device PathState* paths [[buffer(1)]],
   device const HitRecord* hits [[buffer(2)]],
   constant InstanceInfo* instances [[buffer(3)]],
   constant GpuMaterial* materials [[buffer(4)]],
   constant FrameConstants& frame [[buffer(5)]],
   constant uint& bounce [[buffer(6)]],
   uint2 id [[thread_position_in_grid]])
{
    if (id.x >= frame.cam.width || id.y >= frame.cam.height) return;
    uint idx = id.y * frame.cam.width + id.x;

    PathState path = paths[idx];
    if (path.radiance.w == 0) return;

    HitRecord hit = hits[idx];
    if (hit.hit == 0)
    {
        path.radiance.xyz += path.throughput.xyz * frame.envColor.xyz;
        path.radiance.w = 0;
        paths[idx] = path;
        return;
    }

    constant InstanceInfo& info = instances[hit.instanceIndex];
    SurfaceData surf = EvaluateSurface(info, materials, hit,
                                       path.origin.xyz, path.direction.xyz);

    uint rng = as_type<uint>(path.throughput.w);
    float3 wo = -path.direction.xyz;
    Bsdf bsdf = MakeBsdf(surf, frame.debugFlags);

    path.radiance.xyz += path.throughput.xyz * surf.emission;

    // Next event estimation for the directional light
    float3 wl = -frame.lightDir.xyz;
    float nol = dot(surf.normal, wl);
    if (nol > 0 && Luminance(frame.lightColor.xyz) > 0)
    {
        ray shadow;
        shadow.origin = surf.position + surf.normal * kRayEpsilon;
        shadow.direction = wl;
        shadow.min_distance = 0;
        shadow.max_distance = INFINITY;

        intersector<triangle_data, instancing> occ;
        occ.accept_any_intersection(true);
        intersection_result<triangle_data, instancing> sh =
          occ.intersect(shadow, accel, 0xffu);

        if (sh.type == intersection_type::none)
            path.radiance.xyz += path.throughput.xyz *
              EvalBsdf(bsdf, surf.normal, wo, wl) * nol *
              frame.lightColor.xyz;
    }

    // Sample the next bounce
    if (bounce + 1 >= frame.maxBounces)
    {
        path.radiance.w = 0;
    }
    else
    {
        float3 wi;
        float3 weight = SampleBsdf(bsdf, rng, surf.normal, wo, wi);
        if (Luminance(weight) <= 0)
        {
            path.radiance.w = 0;
        }
        else
        {
            path.throughput.xyz *= weight;
            path.origin.xyz = surf.position + surf.normal * kRayEpsilon;
            path.direction.xyz = wi;
        }
    }

    path.throughput.w = as_type<float>(rng);
    paths[idx] = path;
}

static float3 TonemapAces(float3 x)
{
    // Narkowicz ACES approximation
    return saturate(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14));
}

kernel void Resolve
  (device const PathState* paths [[buffer(0)]],
   device float4* accum [[buffer(1)]],
   constant FrameConstants& frame [[buffer(2)]],
   texture2d<float, access::write> output [[texture(0)]],
   uint2 id [[thread_position_in_grid]])
{
    if (id.x >= frame.cam.width || id.y >= frame.cam.height) return;
    uint idx = id.y * frame.cam.width + id.x;

    float4 acc = accum[idx];
    acc.xyz += paths[idx].radiance.xyz;
    acc.w += 1;
    accum[idx] = acc;

    // Output stays linear; Unity's sRGB conversion happens at display time
    // (linear color space project), and the PNG writer encodes explicitly.
    float3 color = acc.xyz / acc.w;
    if (frame.linearOutput == 0)
        color = TonemapAces(color * frame.exposure);
    output.write(float4(color, 1), id);
}

// Probe rays (world space origin/direction float4 pairs); intersection
// regression test path.
kernel void TraceProbes
  (instance_acceleration_structure accel [[buffer(0)]],
   device ProbeResult* output [[buffer(1)]],
   constant float4* rays [[buffer(2)]],
   uint id [[thread_position_in_grid]])
{
    ray r;
    r.origin = rays[id * 2].xyz;
    r.direction = normalize(rays[id * 2 + 1].xyz);
    r.min_distance = 0;
    r.max_distance = INFINITY;

    intersector<triangle_data, instancing> isect;
    intersection_result<triangle_data, instancing> hit =
      isect.intersect(r, accel, 0xffu);

    ProbeResult res = {};
    if (hit.type == intersection_type::triangle)
    {
        res.hit = 1;
        res.distance = hit.distance;
        res.primitiveIndex = hit.primitive_id;
        res.instanceIndex = hit.instance_id;
        res.barycentric = hit.triangle_barycentric_coord;
    }
    output[id] = res;
}
)msl";

void SetError(NSString* message)
{
    s_LastError = message.UTF8String;
    NSLog(@"[MetalRTPlugin] %@", message);
}

bool EnsureDevice()
{
    if (s_Device != nil) return true;

    if (s_Metal == nullptr)
    {
        SetError(@"IUnityGraphicsMetalV1 interface is not available.");
        return false;
    }

    s_Device = s_Metal->MetalDevice();
    if (s_Device == nil)
    {
        SetError(@"Unity Metal device is not ready yet.");
        return false;
    }

    s_Queue = [s_Device newCommandQueue];
    return true;
}

id<MTLComputePipelineState> CreatePipeline(id<MTLLibrary> library,
                                           NSString* name)
{
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil)
    {
        SetError([NSString stringWithFormat:@"Kernel %@ not found.", name]);
        return nil;
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
      [s_Device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil)
        SetError([NSString stringWithFormat:@"Pipeline %@ creation failed: %@",
                  name, error.localizedDescription]);
    return pipeline;
}

bool EnsurePipelines()
{
    if (s_ResolvePipeline != nil) return true;

    NSError* error = nil;
    id<MTLLibrary> library =
      [s_Device newLibraryWithSource:@(kShaderSource)
                             options:nil
                               error:&error];
    if (library == nil)
    {
        SetError([NSString stringWithFormat:@"Shader compilation failed: %@",
                  error.localizedDescription]);
        return false;
    }

    s_ClearPipeline = CreatePipeline(library, @"ClearAccum");
    s_RayGenPipeline = CreatePipeline(library, @"RayGen");
    s_IntersectPipeline = CreatePipeline(library, @"Intersect");
    s_ShadePipeline = CreatePipeline(library, @"Shade");
    s_ResolvePipeline = CreatePipeline(library, @"Resolve");
    s_ProbePipeline = CreatePipeline(library, @"TraceProbes");
    return s_ClearPipeline != nil && s_RayGenPipeline != nil &&
           s_IntersectPipeline != nil && s_ShadePipeline != nil &&
           s_ResolvePipeline != nil && s_ProbePipeline != nil;
}

bool RunCommandBuffer(id<MTLCommandBuffer> command)
{
    [command commit];
    [command waitUntilCompleted];
    if (command.error != nil)
    {
        SetError([NSString stringWithFormat:@"GPU execution failed: %@",
                  command.error.localizedDescription]);
        return false;
    }
    return true;
}

// Fills Metal instance descriptors and the shader-side instance records
// from marshaled instance data. Returns nil on error.
MTLInstanceAccelerationStructureDescriptor* CreateInstanceDescriptor
  (const InstanceDesc* instances, int32_t count, id<MTLBuffer>* outInfo)
{
    id<MTLBuffer> descBuffer = [s_Device
      newBufferWithLength:sizeof(MTLAccelerationStructureInstanceDescriptor)
                          * count
                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> infoBuffer = [s_Device
      newBufferWithLength:sizeof(GpuInstanceInfo) * count
                  options:MTLResourceStorageModeShared];

    auto* descs = (MTLAccelerationStructureInstanceDescriptor*)
      descBuffer.contents;
    auto* infos = (GpuInstanceInfo*)infoBuffer.contents;

    for (int32_t i = 0; i < count; i++)
    {
        const InstanceDesc& src = instances[i];
        if (src.meshIndex < 0 || src.meshIndex >= (int32_t)s_Meshes.size())
        {
            SetError(@"Instance references an invalid mesh index.");
            return nil;
        }
        const MeshEntry& mesh = s_Meshes[src.meshIndex];

        MTLAccelerationStructureInstanceDescriptor& desc = descs[i];
        for (int c = 0; c < 4; c++)
            desc.transformationMatrix.columns[c] =
              MTLPackedFloat3Make(src.objectToWorld[0][c],
                                  src.objectToWorld[1][c],
                                  src.objectToWorld[2][c]);
        desc.options = MTLAccelerationStructureInstanceOptionOpaque;
        desc.mask = 0xff;
        desc.intersectionFunctionTableOffset = 0;
        desc.accelerationStructureIndex = src.meshIndex;

        GpuInstanceInfo& info = infos[i];
        info.vertices = mesh.vertexBuffer.gpuAddress;
        info.indices = mesh.indexBuffer.gpuAddress;
        info.vertexStride = mesh.vertexStride;
        info.positionOffset = mesh.positionOffset;
        info.normalOffset = mesh.normalOffset;
        info.uvOffset = mesh.uvOffset;
        info.indexFormat = mesh.indexFormat;
        info.materialIndex = std::max(src.materialIndex, 0);
        std::memcpy(info.normalMatrix, src.normalMatrix,
                    sizeof(info.normalMatrix));
    }

    NSMutableArray* blases = [NSMutableArray arrayWithCapacity:s_Meshes.size()];
    for (const auto& mesh : s_Meshes) [blases addObject:mesh.blas];

    MTLInstanceAccelerationStructureDescriptor* descriptor =
      [MTLInstanceAccelerationStructureDescriptor descriptor];
    descriptor.instancedAccelerationStructures = blases;
    descriptor.instanceCount = count;
    descriptor.instanceDescriptorBuffer = descBuffer;

    *outInfo = infoBuffer;
    return descriptor;
}

// Encodes an acceleration structure build onto the given command buffer.
id<MTLAccelerationStructure> EncodeAccelerationStructureBuild
  (id<MTLCommandBuffer> command, MTLAccelerationStructureDescriptor* descriptor)
{
    MTLAccelerationStructureSizes sizes =
      [s_Device accelerationStructureSizesWithDescriptor:descriptor];

    id<MTLAccelerationStructure> accel =
      [s_Device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
    id<MTLBuffer> scratch =
      [s_Device newBufferWithLength:sizes.buildScratchBufferSize
                            options:MTLResourceStorageModePrivate];
    if (accel == nil || scratch == nil)
    {
        SetError(@"Acceleration structure allocation failed.");
        return nil;
    }

    id<MTLAccelerationStructureCommandEncoder> encoder =
      [command accelerationStructureCommandEncoder];
    [encoder buildAccelerationStructure:accel
                             descriptor:descriptor
                          scratchBuffer:scratch
                    scratchBufferOffset:0];
    [encoder endEncoding];
    return accel;
}

id<MTLAccelerationStructure> BuildAccelerationStructureSync
  (MTLAccelerationStructureDescriptor* descriptor)
{
    id<MTLCommandBuffer> command = [s_Queue commandBuffer];
    id<MTLAccelerationStructure> accel =
      EncodeAccelerationStructureBuild(command, descriptor);
    if (accel == nil) return nil;
    return RunCommandBuffer(command) ? accel : nil;
}

void EnsurePathBuffers(uint32_t width, uint32_t height)
{
    if (s_BufferWidth == width && s_BufferHeight == height &&
        s_PathBuffer != nil) return;

    NSUInteger pixels = (NSUInteger)width * height;
    s_PathBuffer = [s_Device newBufferWithLength:pixels * 64
                                         options:MTLResourceStorageModePrivate];
    s_HitBuffer = [s_Device newBufferWithLength:pixels * 32
                                        options:MTLResourceStorageModePrivate];
    s_AccumBuffer = [s_Device newBufferWithLength:pixels * 16
                                          options:MTLResourceStorageModePrivate];
    s_BufferWidth = width;
    s_BufferHeight = height;
}

void UseSceneResources(id<MTLComputeCommandEncoder> encoder)
{
    for (const auto& mesh : s_Meshes)
    {
        [encoder useResource:mesh.blas usage:MTLResourceUsageRead];
        [encoder useResource:mesh.vertexBuffer usage:MTLResourceUsageRead];
        [encoder useResource:mesh.indexBuffer usage:MTLResourceUsageRead];
    }
    for (id<MTLTexture> texture : s_MaterialTextures)
        [encoder useResource:texture usage:MTLResourceUsageRead];
}

// Render thread entry point (CommandBuffer.IssuePluginEventAndData).
// Encodes the per-frame TLAS rebuild and the whole path tracing pipeline
// into Unity's current Metal command buffer.
void UNITY_INTERFACE_API OnRenderEvent(int eventId, void* data)
{
    if (s_Metal == nullptr || data == nullptr) return;
    if (!EnsureDevice() || !EnsurePipelines()) return;

    const auto* ev = (const EventData*)data;
    if (ev->instanceCount <= 0 || ev->instanceCount > kMaxEventInstances)
        return;
    if (s_MaterialBuffer == nil) return;

    id<MTLTexture> texture = (__bridge id<MTLTexture>)(void*)ev->texture;
    if (texture == nil || !(texture.usage & MTLTextureUsageShaderWrite))
        return;

    uint32_t width = ev->params.width, height = ev->params.height;
    bool resized = width != s_BufferWidth || height != s_BufferHeight;
    EnsurePathBuffers(width, height);

    id<MTLBuffer> info = nil;
    MTLInstanceAccelerationStructureDescriptor* descriptor =
      CreateInstanceDescriptor(ev->instances, ev->instanceCount, &info);
    if (descriptor == nil) return;

    FrameConstants frame = {};
    frame.cam = ev->params;
    std::memcpy(frame.envColor, ev->envColor, sizeof(float) * 12);
    frame.frameIndex = ev->frameIndex;
    frame.maxBounces = std::min(std::max(ev->maxBounces, 1u), kMaxBounceLimit);
    frame.linearOutput = ev->linearOutput;
    frame.debugFlags = ev->debugFlags;
    frame.exposure = ev->exposure;

    // End Unity's in-flight encoder so we can put our own encoders on its
    // command buffer; Unity opens a new encoder afterwards as needed.
    s_Metal->EndCurrentCommandEncoder();
    id<MTLCommandBuffer> command = s_Metal->CurrentCommandBuffer();
    if (command == nil) return;

    id<MTLAccelerationStructure> tlas =
      EncodeAccelerationStructureBuild(command, descriptor);
    if (tlas == nil) return;

    id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
    UseSceneResources(enc);
    [enc useResource:tlas usage:MTLResourceUsageRead];

    MTLSize grid = MTLSizeMake(width, height, 1);
    MTLSize group = MTLSizeMake(8, 8, 1);

    auto barrier = ^{
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    };

    if (ev->reset != 0 || resized)
    {
        [enc setComputePipelineState:s_ClearPipeline];
        [enc setBuffer:s_AccumBuffer offset:0 atIndex:0];
        [enc setBytes:&frame length:sizeof(frame) atIndex:1];
        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        barrier();
    }

    [enc setComputePipelineState:s_RayGenPipeline];
    [enc setBuffer:s_PathBuffer offset:0 atIndex:0];
    [enc setBytes:&frame length:sizeof(frame) atIndex:1];
    [enc dispatchThreads:grid threadsPerThreadgroup:group];

    for (uint32_t bounce = 0; bounce < frame.maxBounces; bounce++)
    {
        barrier();
        [enc setComputePipelineState:s_IntersectPipeline];
        [enc setAccelerationStructure:tlas atBufferIndex:0];
        [enc setBuffer:s_PathBuffer offset:0 atIndex:1];
        [enc setBuffer:s_HitBuffer offset:0 atIndex:2];
        [enc setBytes:&frame length:sizeof(frame) atIndex:3];
        [enc dispatchThreads:grid threadsPerThreadgroup:group];

        barrier();
        [enc setComputePipelineState:s_ShadePipeline];
        [enc setAccelerationStructure:tlas atBufferIndex:0];
        [enc setBuffer:s_PathBuffer offset:0 atIndex:1];
        [enc setBuffer:s_HitBuffer offset:0 atIndex:2];
        [enc setBuffer:info offset:0 atIndex:3];
        [enc setBuffer:s_MaterialBuffer offset:0 atIndex:4];
        [enc setBytes:&frame length:sizeof(frame) atIndex:5];
        [enc setBytes:&bounce length:sizeof(bounce) atIndex:6];
        [enc dispatchThreads:grid threadsPerThreadgroup:group];
    }

    barrier();
    [enc setComputePipelineState:s_ResolvePipeline];
    [enc setBuffer:s_PathBuffer offset:0 atIndex:0];
    [enc setBuffer:s_AccumBuffer offset:0 atIndex:1];
    [enc setBytes:&frame length:sizeof(frame) atIndex:2];
    [enc setTexture:texture atIndex:0];
    [enc dispatchThreads:grid threadsPerThreadgroup:group];
    [enc endEncoding];

    s_EventFrames.fetch_add(1, std::memory_order_relaxed);
}

} // anonymous namespace

extern "C" {

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
UnityPluginLoad(IUnityInterfaces* interfaces)
{
    s_Metal = interfaces->Get<IUnityGraphicsMetalV1>();
}

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API UnityPluginUnload()
{
    s_Metal = nullptr;
}

const char* UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
MetalRT_GetLastError()
{
    return s_LastError.c_str();
}

// -1: device unavailable, 0: no ray tracing support, 1: supported
int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
MetalRT_DeviceSupportsRaytracing()
{
    if (!EnsureDevice()) return -1;
    return s_Device.supportsRaytracing ? 1 : 0;
}

// Registers a mesh and builds its BLAS synchronously. Returns the mesh
// index (>= 0) on success or a negative error code. Attribute offsets are
// byte offsets in the vertex buffer; pass 0xffffffff for absent attributes.
int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
MetalRT_AddMesh
  (void* vertexBuffer, uint32_t vertexStride, uint32_t positionOffset,
   uint32_t normalOffset, uint32_t uvOffset,
   void* indexBuffer, uint32_t indexFormat, uint32_t indexByteOffset,
   uint32_t triangleCount)
{
    if (!EnsureDevice()) return -1;

    if (!s_Device.supportsRaytracing)
    {
        SetError(@"This Metal device does not support ray tracing.");
        return -2;
    }

    MeshEntry mesh = {};
    mesh.vertexBuffer = (__bridge id<MTLBuffer>)vertexBuffer;
    mesh.indexBuffer = (__bridge id<MTLBuffer>)indexBuffer;
    mesh.vertexStride = vertexStride;
    mesh.positionOffset = positionOffset;
    mesh.normalOffset = normalOffset;
    mesh.uvOffset = uvOffset;
    mesh.indexFormat = indexFormat;

    MTLAccelerationStructureTriangleGeometryDescriptor* geometry =
      [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    geometry.vertexBuffer = mesh.vertexBuffer;
    geometry.vertexBufferOffset = positionOffset;
    geometry.vertexStride = vertexStride;
    geometry.vertexFormat = MTLAttributeFormatFloat3;
    geometry.indexBuffer = mesh.indexBuffer;
    geometry.indexBufferOffset = indexByteOffset;
    geometry.indexType =
      indexFormat == 0 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
    geometry.triangleCount = triangleCount;
    geometry.opaque = YES;

    MTLPrimitiveAccelerationStructureDescriptor* descriptor =
      [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    descriptor.geometryDescriptors = @[geometry];

    mesh.blas = BuildAccelerationStructureSync(descriptor);
    if (mesh.blas == nil) return -3;

    s_Meshes.push_back(mesh);
    return (int)s_Meshes.size() - 1;
}

// Builds the material table. Base maps are referenced bindlessly via
// MTLResourceID, so the textures must stay alive while tracing.
int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
MetalRT_SetMaterials(const MaterialDesc* materials, int32_t count)
{
    if (!EnsureDevice()) return -1;
    if (count <= 0) return -2;

    id<MTLBuffer> buffer = [s_Device
      newBufferWithLength:sizeof(GpuMaterial) * count
                  options:MTLResourceStorageModeShared];
    auto* dst = (GpuMaterial*)buffer.contents;

    s_MaterialTextures.clear();

    for (int32_t i = 0; i < count; i++)
    {
        const MaterialDesc& src = materials[i];
        GpuMaterial& mat = dst[i];
        std::memcpy(mat.baseColor, src.baseColor, sizeof(float) * 12);
        mat.metallic = src.metallic;
        mat.smoothness = src.smoothness;
        mat.hasBaseMap = 0;
        if (src.hasBaseMap != 0 && src.texture != 0)
        {
            id<MTLTexture> texture =
              (__bridge id<MTLTexture>)(void*)src.texture;
            mat.baseMap = texture.gpuResourceID;
            mat.hasBaseMap = 1;
            s_MaterialTextures.push_back(texture);
        }
    }

    s_MaterialBuffer = buffer;
    return 0;
}

// Synchronously (re)builds the instance acceleration structure used by the
// probe test path.
int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
MetalRT_BuildInstanceAS(const InstanceDesc* instances, int32_t count)
{
    if (!EnsureDevice()) return -1;

    if (s_Meshes.empty() || count <= 0)
    {
        SetError(@"No meshes registered or empty instance list.");
        return -2;
    }

    id<MTLBuffer> info = nil;
    MTLInstanceAccelerationStructureDescriptor* descriptor =
      CreateInstanceDescriptor(instances, count, &info);
    if (descriptor == nil) return -3;

    s_InstanceAS = BuildAccelerationStructureSync(descriptor);
    if (s_InstanceAS == nil) return -4;

    s_InstanceInfo = info;
    return 0;
}

// rays: (origin.xyzw, direction.xyzw) float4 pairs in world space
int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
MetalRT_TraceProbes(const float* rays, int32_t count, ProbeResult* results)
{
    if (s_InstanceAS == nil)
    {
        SetError(@"Instance acceleration structure has not been built.");
        return -1;
    }
    if (!EnsurePipelines()) return -2;

    NSUInteger size = sizeof(ProbeResult) * count;
    id<MTLBuffer> output =
      [s_Device newBufferWithLength:size options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> command = [s_Queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:s_ProbePipeline];
    [encoder setAccelerationStructure:s_InstanceAS atBufferIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBytes:rays length:sizeof(float) * 8 * count atIndex:2];
    for (const auto& mesh : s_Meshes)
        [encoder useResource:mesh.blas usage:MTLResourceUsageRead];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];

    if (!RunCommandBuffer(command)) return -3;

    std::memcpy(results, output.contents, size);
    return 0;
}

// Returns the render event callback for CommandBuffer.IssuePluginEventAndData.
UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
MetalRT_GetRenderEventFunc()
{
    return OnRenderEvent;
}

// Number of render events executed so far (verifies the render thread path).
int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
MetalRT_GetEventFrameCount()
{
    return s_EventFrames.load(std::memory_order_relaxed);
}

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API MetalRT_Dispose()
{
    s_InstanceAS = nil;
    s_InstanceInfo = nil;
    s_MaterialBuffer = nil;
    s_MaterialTextures.clear();
    s_Meshes.clear();
    s_PathBuffer = nil;
    s_HitBuffer = nil;
    s_AccumBuffer = nil;
    s_BufferWidth = s_BufferHeight = 0;
    s_EventFrames.store(0, std::memory_order_relaxed);
}

} // extern "C"
