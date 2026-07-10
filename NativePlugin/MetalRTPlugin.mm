// Metal hardware ray tracing path tracer plugin for Unity (stage 2:
// wavefront material evaluation with Unity-compiled compute shaders)
//
// Builds per-mesh primitive acceleration structures (BLAS) directly from
// Unity mesh GPU buffers, combines them into an instance acceleration
// structure (TLAS), and path-traces the scene with a wavefront-style
// per-bounce kernel pipeline. The per-frame pipeline is split into phases
// (Begin / Intersect / Shade / Resolve) driven by separate plugin render
// events, so Unity-side compute dispatches (Shader Graph material
// evaluation) can be interleaved between Intersect and Shade on the same
// command stream:
//
//   Begin: TLAS rebuild, accumulation clear, RayGen
//   per bounce:
//     Intersect: TLAS intersection -> hit records (shared buffer)
//                GeomPrep: attribute interpolation -> hit attributes
//                          + default URP Lit surface evaluation
//     (Unity dispatches material evaluation compute shaders here,
//      overwriting surface records for their material indices)
//     Shade: NEE + BSDF sampling from the surface records
//   Resolve: progressive accumulation + tonemap to the output texture
//
// Hit records, hit attributes, and surface records live in Unity-created
// GraphicsBuffers shared with this plugin by their native pointers, so both
// the native kernels and Unity compute shaders can access them.

#import <Metal/Metal.h>

#include "IUnityInterface.h"
#include "IUnityGraphics.h"
#include "IUnityGraphicsMetal.h"

#include <atomic>
#include <cstring>
#include <string>
#include <vector>

namespace {

IUnityGraphicsMetalV2* s_Metal;

id<MTLDevice> s_Device;
id<MTLCommandQueue> s_Queue;      // private queue (synchronous setup path)
id<MTLCommandQueue> s_UnityQueue; // Unity's queue (render thread phases)

id<MTLComputePipelineState> s_ClearPipeline;
id<MTLComputePipelineState> s_RayGenPipeline;
id<MTLComputePipelineState> s_IntersectPipeline;
id<MTLComputePipelineState> s_GeomPrepPipeline;
id<MTLComputePipelineState> s_ShadePipeline;
id<MTLComputePipelineState> s_ResolvePipeline;
id<MTLComputePipelineState> s_AtrousPipeline;
id<MTLComputePipelineState> s_OutputPipeline;
id<MTLComputePipelineState> s_ProbePipeline;

struct MeshEntry
{
    id<MTLAccelerationStructure> blas;
    id<MTLBuffer> vertexBuffer;
    id<MTLBuffer> indexBuffer;
    uint32_t vertexStride, positionOffset, normalOffset, tangentOffset;
    uint32_t uvOffset, uv1Offset, colorOffset, colorFormat;
    uint32_t indexFormat;
};

// Written on the main thread during setup (before any render event fires),
// read from the render thread afterwards.
std::vector<MeshEntry> s_Meshes;
id<MTLBuffer> s_MaterialBuffer;
std::vector<id<MTLTexture>> s_MaterialTextures;
id<MTLAccelerationStructure> s_InstanceAS; // synchronous path only
id<MTLBuffer> s_InstanceInfo;              // synchronous path only

// Shared with Unity (GraphicsBuffer native pointers)
id<MTLBuffer> s_SharedHits;
id<MTLBuffer> s_SharedAttributes;
id<MTLBuffer> s_SharedSurfaces;

// Native-only per-frame resources (render thread)
id<MTLBuffer> s_PathBuffer;
id<MTLBuffer> s_AccumBuffer;
id<MTLBuffer> s_GBuffer;   // primary hit albedo + normal (denoiser guides)
id<MTLBuffer> s_DenoiseA;  // irradiance ping-pong buffers
id<MTLBuffer> s_DenoiseB;
uint32_t s_BufferWidth, s_BufferHeight;
uint32_t s_AccumFrames; // frames since the last accumulation reset

constexpr uint32_t kDenoiseFrameLimit = 64; // stop filtering once converged
constexpr int kDenoiseIterations = 3;

// Per-frame state carried across the phase events of one frame
// (render thread only).
id<MTLAccelerationStructure> s_FrameTlas;
id<MTLBuffer> s_FrameInfo;
id<MTLTexture> s_FrameTexture;

std::atomic<int> s_EventFrames {0};

std::string s_LastError;

constexpr uint32_t kNoAttribute = 0xffffffffu;
constexpr int kMaxEventInstances = 16;
constexpr uint32_t kMaxBounceLimit = 8;

// Render event phases; must match MetalRTPlugin.cs
// (eventId = phase | bounce << 8).
enum EventPhase
{
    kPhaseBegin = 0,
    kPhaseIntersect = 1,
    kPhaseShade = 2,
    kPhaseResolve = 3,
};

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
    float cutoff;       // alpha clip threshold; < 0 disables clipping
    uint32_t hasBaseMap;
    uint64_t texture;   // MTLTexture pointer or 0
    uint64_t pad2;
};

// Debug flags in EventData (must match PathTracerTest.cs).
enum DebugFlags
{
    kDebugDiffuseOnly = 1, // disable the specular lobe (analytic tests)
};

constexpr int kMaxLights = 4;

// Must match LightDesc in MetalRTPlugin.cs and in the shader source.
struct LightDesc
{
    float position[4];  // xyz; w: type (0: directional, 1: point, 2: spot)
    float direction[4]; // xyz: direction the light travels; w: range
    float color[4];     // linear color * intensity * pi
    float spot[4];      // x: angle scale, y: angle offset (URP convention)
};

// Must match the event data blob layout in MetalRTPlugin.cs.
struct EventData
{
    TraceParams params;    // 0
    uint64_t texture;      // 80: RenderTexture native pointer
    int32_t instanceCount; // 88
    uint32_t frameIndex;   // 92
    float envColor[4];     // 96: linear radiance of the uniform environment
    uint32_t reset;        // 112
    uint32_t maxBounces;   // 116
    uint32_t linearOutput; // 120: 1 = skip tonemap (analytic tests)
    uint32_t debugFlags;   // 124
    float exposure;        // 128
    uint32_t lightCount;   // 132
    float pad[2];          // 136..144
    LightDesc lights[kMaxLights];            // 144..400
    InstanceDesc instances[kMaxEventInstances];
};

// Shader-side per-instance record; must match InstanceInfo in the shader.
struct GpuInstanceInfo
{
    uint64_t vertices, indices;
    uint32_t vertexStride, positionOffset, normalOffset, tangentOffset;
    uint32_t uvOffset, uv1Offset, colorOffset, colorFormat;
    uint32_t indexFormat, materialIndex, pad0, pad1;
    float objectToWorld[3][4];
    float normalMatrix[3][4];
};

// Shader-side material record; must match GpuMaterial in the shader.
// The base map is referenced by MTLResourceID (Metal 3 bindless).
struct GpuMaterial
{
    float baseColor[4];
    float emission[4];
    float baseMapST[4];
    float metallic, smoothness, cutoff;
    uint32_t hasBaseMap;
    MTLResourceID baseMap;
    uint64_t pad2;
};

// Shader-side per-frame constants; must match FrameConstants in the shader.
struct FrameConstants
{
    TraceParams cam;
    float envColor[4];
    uint32_t frameIndex, maxBounces, linearOutput, debugFlags;
    float exposure;
    uint32_t lightCount;
    uint32_t pad[2];
    LightDesc lights[kMaxLights];
};

FrameConstants s_Frame; // set at kPhaseBegin, reused by later phases

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

struct LightDesc
{
    float4 position;  // xyz; w: type (0: directional, 1: point, 2: spot)
    float4 direction; // xyz: direction the light travels; w: range
    float4 color;     // linear color * intensity * pi
    float4 spot;      // x: angle scale, y: angle offset
};

struct FrameConstants
{
    TraceParams cam;
    float4 envColor;
    uint frameIndex, maxBounces, linearOutput, debugFlags;
    float exposure;
    uint lightCount;
    uint pad0, pad1;
    LightDesc lights[4];
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
    uint vertexStride, positionOffset, normalOffset, tangentOffset;
    uint uvOffset, uv1Offset, colorOffset, colorFormat;
    uint indexFormat, materialIndex, pad0, pad1;
    float4 objectToWorld0, objectToWorld1, objectToWorld2;
    float4 normalMatrix0, normalMatrix1, normalMatrix2;
};

struct GpuMaterial
{
    float4 baseColor;
    float4 emission;
    float4 baseMapST;
    float metallic, smoothness, cutoff;
    uint hasBaseMap;
    texture2d<float> baseMap;
    ulong pad2;
};

// Per-pixel path state (w components carry rng state / alive flag).
struct PathState
{
    float4 origin;
    float4 direction;
    float4 throughput; // w: rng state (as_type)
    float4 radiance;   // w: alive flag
};

// Shared with Unity compute shaders; must match the HLSL declarations.
struct HitRecord
{
    float distance;
    uint instanceIndex, primitiveIndex, hit;
    float2 barycentric, pad;
};

struct HitAttributes
{
    float4 position; // xyz: world position
    float4 normal;   // xyz: world shading normal (faces the ray origin)
    float4 tangent;  // xyz: world tangent, w: bitangent sign
    float4 uvView;   // xy: uv0, zw: uv1
    float4 viewDir;  // xyz: direction toward the ray origin
    float4 color;    // vertex color ((1,1,1,1) when absent)
    uint4 meta;      // x: material index, y: hit flag
};

struct SurfaceRecord
{
    float4 baseColor; // rgb: linear albedo
    float4 normal;    // xyz: world shading normal
    float4 emission;  // rgb: linear radiance
    float4 params;    // x: metallic, y: smoothness,
                      // z: alpha, w: clip threshold (< 0: no clipping)
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

static float4 LoadFloat4Attr
  (constant InstanceInfo& info, uint offset, uint index)
{
    device const packed_float4* p = (device const packed_float4*)
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

// --- BSDF: URP Lit compatible metallic-roughness GGX + Lambert ---

struct Bsdf
{
    float3 diffuse; // Lambert albedo
    float3 f0;      // specular reflectance at normal incidence
    float alpha;    // GGX alpha (perceptual roughness squared)
};

static Bsdf MakeBsdf(float3 baseColor, float metallic, float smoothness,
                     uint debugFlags)
{
    Bsdf b;
    b.diffuse = baseColor * (1 - metallic);
    b.f0 = mix(float3(0.04), baseColor, metallic);
    float perceptual = 1 - smoothness;
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
    return float3x3(t, bt, n);
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

// Interpolates hit point attributes and evaluates the default URP Lit
// material into the surface record. Unity-side material evaluation compute
// shaders may overwrite surface records afterwards (before Shade).
kernel void GeomPrep
  (device const PathState* paths [[buffer(0)]],
   device const HitRecord* hits [[buffer(1)]],
   constant InstanceInfo* instances [[buffer(2)]],
   constant GpuMaterial* materials [[buffer(3)]],
   constant FrameConstants& frame [[buffer(4)]],
   device HitAttributes* attributes [[buffer(5)]],
   device SurfaceRecord* surfaces [[buffer(6)]],
   uint2 id [[thread_position_in_grid]])
{
    if (id.x >= frame.cam.width || id.y >= frame.cam.height) return;
    uint idx = id.y * frame.cam.width + id.x;

    HitAttributes attr = {};
    SurfaceRecord surf = {};

    PathState path = paths[idx];
    HitRecord hit = hits[idx];

    if (path.radiance.w != 0 && hit.hit != 0)
    {
        constant InstanceInfo& info = instances[hit.instanceIndex];
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

        n = float3(dot(info.normalMatrix0.xyz, n),
                   dot(info.normalMatrix1.xyz, n),
                   dot(info.normalMatrix2.xyz, n));
        n = normalize(n);
        if (dot(n, path.direction.xyz) > 0) n = -n;

        // Tangent (world space)
        float4 tangent = float4(1, 0, 0, 1);
        if (info.tangentOffset != 0xffffffffu)
        {
            float4 t0 = LoadFloat4Attr(info, info.tangentOffset, tri.x);
            float4 t1 = LoadFloat4Attr(info, info.tangentOffset, tri.y);
            float4 t2 = LoadFloat4Attr(info, info.tangentOffset, tri.z);
            float4 ts = t0 * bary.x + t1 * bary.y + t2 * bary.z;
            float3 tw = float3(dot(info.objectToWorld0.xyz, ts.xyz),
                               dot(info.objectToWorld1.xyz, ts.xyz),
                               dot(info.objectToWorld2.xyz, ts.xyz));
            tangent = float4(normalize(tw), ts.w);
        }

        // UV channels
        float2 uv = float2(0);
        if (info.uvOffset != 0xffffffffu)
        {
            float2 a = LoadFloat2Attr(info, info.uvOffset, tri.x);
            float2 b = LoadFloat2Attr(info, info.uvOffset, tri.y);
            float2 c = LoadFloat2Attr(info, info.uvOffset, tri.z);
            uv = a * bary.x + b * bary.y + c * bary.z;
        }
        float2 uv1 = float2(0);
        if (info.uv1Offset != 0xffffffffu)
        {
            float2 a = LoadFloat2Attr(info, info.uv1Offset, tri.x);
            float2 b = LoadFloat2Attr(info, info.uv1Offset, tri.y);
            float2 c = LoadFloat2Attr(info, info.uv1Offset, tri.z);
            uv1 = a * bary.x + b * bary.y + c * bary.z;
        }

        // Vertex color (Float32x4 or UNorm8x4)
        float4 color = float4(1);
        if (info.colorOffset != 0xffffffffu)
        {
            float4 c0, c1, c2;
            if (info.colorFormat == 0)
            {
                c0 = LoadFloat4Attr(info, info.colorOffset, tri.x);
                c1 = LoadFloat4Attr(info, info.colorOffset, tri.y);
                c2 = LoadFloat4Attr(info, info.colorOffset, tri.z);
            }
            else
            {
                device const uchar* p = info.vertices + info.colorOffset;
                uint s = info.vertexStride;
                c0 = float4(*(device const uchar4*)(p + s * tri.x)) / 255;
                c1 = float4(*(device const uchar4*)(p + s * tri.y)) / 255;
                c2 = float4(*(device const uchar4*)(p + s * tri.z)) / 255;
            }
            color = c0 * bary.x + c1 * bary.y + c2 * bary.z;
        }

        attr.position = float4(path.origin.xyz +
                               path.direction.xyz * hit.distance, 1);
        attr.normal = float4(n, 0);
        attr.tangent = tangent;
        attr.uvView = float4(uv, uv1);
        attr.viewDir = float4(-path.direction.xyz, 0);
        attr.color = color;
        attr.meta = uint4(info.materialIndex, 1, 0, 0);

        // Default URP Lit surface evaluation
        constant GpuMaterial& mat = materials[info.materialIndex];
        float4 baseColor = mat.baseColor;
        if (mat.hasBaseMap != 0 && info.uvOffset != 0xffffffffu)
        {
            float2 st = uv * mat.baseMapST.xy + mat.baseMapST.zw;
            constexpr sampler smp(address::repeat, filter::linear,
                                  mip_filter::linear);
            baseColor *= mat.baseMap.sample(smp, st, level(2));
        }
        surf.baseColor = float4(baseColor.xyz, 1);
        surf.normal = float4(n, 0);
        surf.emission = mat.emission;
        surf.params = float4(mat.metallic, mat.smoothness,
                             baseColor.w, mat.cutoff);
    }

    attributes[idx] = attr;
    surfaces[idx] = surf;
}

// Shadow ray visibility with alpha-test transparency for materials in the
// native Lit table (cutoff >= 0). Materials evaluated by Unity compute
// shaders keep their table cutoff (usually opaque). Up to 4 transparent
// layers; more counts as occluded.
static bool ShadowVisible
  (instance_acceleration_structure accel,
   constant InstanceInfo* instances,
   constant GpuMaterial* materials,
   float3 origin, float3 dir, float maxDist)
{
    float minDist = 0;
    for (int iter = 0; iter < 4; iter++)
    {
        ray r;
        r.origin = origin;
        r.direction = dir;
        r.min_distance = minDist;
        r.max_distance = maxDist;

        intersector<triangle_data, instancing> isect;
        intersection_result<triangle_data, instancing> hit =
          isect.intersect(r, accel, 0xffu);
        if (hit.type == intersection_type::none) return true;

        constant InstanceInfo& info = instances[hit.instance_id];
        constant GpuMaterial& mat = materials[info.materialIndex];
        if (mat.cutoff < 0) return false; // opaque

        float alpha = mat.baseColor.w;
        if (mat.hasBaseMap != 0 && info.uvOffset != 0xffffffffu)
        {
            uint3 tri = LoadTriangleIndices(info, hit.primitive_id);
            float3 bary = float3(1 - hit.triangle_barycentric_coord.x -
                                 hit.triangle_barycentric_coord.y,
                                 hit.triangle_barycentric_coord.x,
                                 hit.triangle_barycentric_coord.y);
            float2 a = LoadFloat2Attr(info, info.uvOffset, tri.x);
            float2 b = LoadFloat2Attr(info, info.uvOffset, tri.y);
            float2 c = LoadFloat2Attr(info, info.uvOffset, tri.z);
            float2 uv = a * bary.x + b * bary.y + c * bary.z;
            float2 st = uv * mat.baseMapST.xy + mat.baseMapST.zw;
            constexpr sampler smp(address::repeat, filter::linear,
                                  mip_filter::linear);
            alpha *= mat.baseMap.sample(smp, st, level(2)).w;
        }
        if (alpha >= mat.cutoff) return false;

        minDist = hit.distance + kRayEpsilon; // pass through
    }
    return false;
}

kernel void Shade
  (instance_acceleration_structure accel [[buffer(0)]],
   device PathState* paths [[buffer(1)]],
   device const HitAttributes* attributes [[buffer(2)]],
   device const SurfaceRecord* surfaces [[buffer(3)]],
   constant FrameConstants& frame [[buffer(4)]],
   constant uint& bounce [[buffer(5)]],
   constant InstanceInfo* instances [[buffer(6)]],
   constant GpuMaterial* materials [[buffer(7)]],
   device float4* gbuffer [[buffer(8)]],
   uint2 id [[thread_position_in_grid]])
{
    if (id.x >= frame.cam.width || id.y >= frame.cam.height) return;
    uint idx = id.y * frame.cam.width + id.x;

    PathState path = paths[idx];
    if (path.radiance.w == 0) return;

    HitAttributes attr = attributes[idx];
    if (attr.meta.y == 0)
    {
        if (bounce == 0) // denoiser guides: environment pixel
        {
            gbuffer[idx * 2] = float4(1, 1, 1, 1);
            gbuffer[idx * 2 + 1] = float4(0);
        }
        path.radiance.xyz += path.throughput.xyz * frame.envColor.xyz;
        path.radiance.w = 0;
        paths[idx] = path;
        return;
    }

    SurfaceRecord surf = surfaces[idx];
    float3 position = attr.position.xyz;

    // Alpha clipped hit: pass through (continue the ray past the surface;
    // this consumes one bounce iteration). Shadow rays evaluate the alpha
    // of native Lit materials; compute-evaluated clipping stays opaque.
    if (surf.params.w >= 0 && surf.params.z < surf.params.w)
    {
        if (bounce == 0)
        {
            gbuffer[idx * 2] = float4(1, 1, 1, 1);
            gbuffer[idx * 2 + 1] = float4(0);
        }
        path.origin.xyz = position + path.direction.xyz * kRayEpsilon;
        paths[idx] = path;
        return;
    }

    float3 normal = normalize(surf.normal.xyz);

    if (bounce == 0) // denoiser guides: primary hit albedo and normal
    {
        gbuffer[idx * 2] = float4(surf.baseColor.xyz, 1);
        gbuffer[idx * 2 + 1] = float4(normal, 0);
    }

    uint rng = as_type<uint>(path.throughput.w);
    float3 wo = -path.direction.xyz;
    Bsdf bsdf = MakeBsdf(surf.baseColor.xyz, surf.params.x, surf.params.y,
                         frame.debugFlags);

    path.radiance.xyz += path.throughput.xyz * surf.emission.xyz;

    // Next event estimation for punctual lights (directional/point/spot,
    // URP attenuation conventions)
    for (uint li = 0; li < frame.lightCount; li++)
    {
        LightDesc light = frame.lights[li];
        int type = (int)light.position.w;

        float3 wl;
        float shadowMax = INFINITY;
        float atten = 1;
        if (type == 0)
        {
            wl = -light.direction.xyz;
        }
        else
        {
            float3 dv = light.position.xyz - position;
            float d2 = max(dot(dv, dv), 1e-6);
            float dist = sqrt(d2);
            wl = dv / dist;
            shadowMax = dist - kRayEpsilon;

            // URP distance attenuation: 1/d^2 with smooth range window
            float r2 = light.direction.w * light.direction.w;
            float window = saturate(1 - (d2 / r2) * (d2 / r2));
            atten = window * window / d2;

            if (type == 2) // spot angle attenuation
            {
                float cd = dot(-wl, normalize(light.direction.xyz));
                float a = saturate(cd * light.spot.x + light.spot.y);
                atten *= a * a;
            }
        }

        float nol = dot(normal, wl);
        if (nol <= 0 || atten <= 0 ||
            Luminance(light.color.xyz) <= 0) continue;

        if (ShadowVisible(accel, instances, materials,
                          position + normal * kRayEpsilon, wl, shadowMax))
            path.radiance.xyz += path.throughput.xyz *
              EvalBsdf(bsdf, normal, wo, wl) * nol *
              light.color.xyz * atten;
    }

    // Sample the next bounce
    if (bounce + 1 >= frame.maxBounces)
    {
        path.radiance.w = 0;
    }
    else
    {
        float3 wi;
        float3 weight = SampleBsdf(bsdf, rng, normal, wo, wi);
        if (Luminance(weight) <= 0)
        {
            path.radiance.w = 0;
        }
        else
        {
            path.throughput.xyz *= weight;
            path.origin.xyz = position + normal * kRayEpsilon;
            path.direction.xyz = wi;
        }
    }

    path.throughput.w = as_type<float>(rng);
    paths[idx] = path;
}

static float3 TonemapAces(float3 x)
{
    return saturate(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14));
}

// Accumulates the frame and writes the albedo-demodulated irradiance mean
// (the denoiser filters irradiance so texture detail is preserved).
kernel void Resolve
  (device const PathState* paths [[buffer(0)]],
   device float4* accum [[buffer(1)]],
   constant FrameConstants& frame [[buffer(2)]],
   device const float4* gbuffer [[buffer(3)]],
   device float4* irradiance [[buffer(4)]],
   uint2 id [[thread_position_in_grid]])
{
    if (id.x >= frame.cam.width || id.y >= frame.cam.height) return;
    uint idx = id.y * frame.cam.width + id.x;

    float4 acc = accum[idx];
    acc.xyz += paths[idx].radiance.xyz;
    acc.w += 1;
    accum[idx] = acc;

    float3 mean = acc.xyz / acc.w;
    float3 albedo = max(gbuffer[idx * 2].xyz, 0.05);
    irradiance[idx] = float4(mean / albedo, 1);
}

// Edge-avoiding a-trous wavelet filter step over the irradiance buffer,
// guided by the primary hit albedo and normal.
kernel void Atrous
  (device const float4* source [[buffer(0)]],
   device float4* destination [[buffer(1)]],
   device const float4* gbuffer [[buffer(2)]],
   constant FrameConstants& frame [[buffer(3)]],
   constant uint& step [[buffer(4)]],
   uint2 id [[thread_position_in_grid]])
{
    uint w = frame.cam.width, h = frame.cam.height;
    if (id.x >= w || id.y >= h) return;
    uint idx = id.y * w + id.x;

    constexpr float taps[5] = { 1.0 / 16, 1.0 / 4, 3.0 / 8, 1.0 / 4,
                                1.0 / 16 };

    float3 albedo = gbuffer[idx * 2].xyz;
    float3 normal = gbuffer[idx * 2 + 1].xyz;
    float3 center = source[idx].xyz;

    float3 sum = 0;
    float weightSum = 0;
    for (int dy = -2; dy <= 2; dy++)
    for (int dx = -2; dx <= 2; dx++)
    {
        int2 q = clamp(int2(id) + int2(dx, dy) * (int)step,
                       int2(0), int2(w - 1, h - 1));
        uint qi = q.y * w + q.x;
        float weight = taps[dx + 2] * taps[dy + 2];
        if (dx != 0 || dy != 0)
        {
            float3 qn = gbuffer[qi * 2 + 1].xyz;
            float3 da = gbuffer[qi * 2].xyz - albedo;
            float3 dc = source[qi].xyz - center;
            weight *= pow(max(dot(normal, qn), 0.0), 32.0);
            weight *= exp(-dot(da, da) / 0.02);
            weight *= exp(-dot(dc, dc) / 0.25);
        }
        sum += source[qi].xyz * weight;
        weightSum += weight;
    }
    destination[idx] = float4(sum / max(weightSum, 1e-6), 1);
}

// Remodulates the (possibly filtered) irradiance with the albedo and
// writes the display value into the output texture.
kernel void Output
  (device const float4* irradiance [[buffer(0)]],
   device const float4* gbuffer [[buffer(1)]],
   constant FrameConstants& frame [[buffer(2)]],
   texture2d<float, access::write> output [[texture(0)]],
   uint2 id [[thread_position_in_grid]])
{
    if (id.x >= frame.cam.width || id.y >= frame.cam.height) return;
    uint idx = id.y * frame.cam.width + id.x;

    float3 albedo = max(gbuffer[idx * 2].xyz, 0.05);
    float3 color = irradiance[idx].xyz * albedo;

    // Output stays linear; Unity's sRGB conversion happens at display time.
    if (frame.linearOutput == 0)
        color = TonemapAces(color * frame.exposure);
    output.write(float4(color, 1), id);
}

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
        SetError(@"IUnityGraphicsMetalV2 interface is not available.");
        return false;
    }

    s_Device = s_Metal->MetalDevice();
    if (s_Device == nil)
    {
        SetError(@"Unity Metal device is not ready yet.");
        return false;
    }

    s_Queue = [s_Device newCommandQueue];
    s_UnityQueue = s_Metal->CommandQueue();
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
    s_GeomPrepPipeline = CreatePipeline(library, @"GeomPrep");
    s_ShadePipeline = CreatePipeline(library, @"Shade");
    s_ResolvePipeline = CreatePipeline(library, @"Resolve");
    s_AtrousPipeline = CreatePipeline(library, @"Atrous");
    s_OutputPipeline = CreatePipeline(library, @"Output");
    s_ProbePipeline = CreatePipeline(library, @"TraceProbes");
    return s_ClearPipeline != nil && s_RayGenPipeline != nil &&
           s_IntersectPipeline != nil && s_GeomPrepPipeline != nil &&
           s_ShadePipeline != nil && s_ResolvePipeline != nil &&
           s_AtrousPipeline != nil && s_OutputPipeline != nil &&
           s_ProbePipeline != nil;
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
        info.tangentOffset = mesh.tangentOffset;
        info.uvOffset = mesh.uvOffset;
        info.uv1Offset = mesh.uv1Offset;
        info.colorOffset = mesh.colorOffset;
        info.colorFormat = mesh.colorFormat;
        info.indexFormat = mesh.indexFormat;
        info.materialIndex = std::max(src.materialIndex, 0);
        std::memcpy(info.objectToWorld, src.objectToWorld,
                    sizeof(info.objectToWorld));
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
    s_AccumBuffer = [s_Device newBufferWithLength:pixels * 16
                                          options:MTLResourceStorageModePrivate];
    s_GBuffer = [s_Device newBufferWithLength:pixels * 32
                                      options:MTLResourceStorageModePrivate];
    s_DenoiseA = [s_Device newBufferWithLength:pixels * 16
                                       options:MTLResourceStorageModePrivate];
    s_DenoiseB = [s_Device newBufferWithLength:pixels * 16
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

// Commits Unity's current command buffer (so everything Unity encoded so
// far — including material evaluation dispatches — is ordered before us)
// and returns a fresh command buffer of our own on Unity's queue. The queue
// executes command buffers in commit order, so the phase pipeline stays
// GPU-ordered without ever touching Unity's encoder state.
id<MTLCommandBuffer> BeginPhaseCommandBuffer()
{
    s_Metal->CommitCurrentCommandBuffer();
    return [s_UnityQueue commandBuffer];
}

// Render thread entry point (CommandBuffer.IssuePluginEventAndData).
// eventId = phase | bounce << 8. The phases of one frame are issued as
// separate events so Unity compute dispatches (material evaluation) can be
// interleaved between Intersect and Shade on the same command stream.
void UNITY_INTERFACE_API OnRenderEvent(int eventId, void* data)
{
    if (s_Metal == nullptr || data == nullptr) return;
    if (!EnsureDevice() || !EnsurePipelines()) return;

    if (s_SharedHits == nil || s_SharedAttributes == nil ||
        s_SharedSurfaces == nil || s_MaterialBuffer == nil) return;

    const auto* ev = (const EventData*)data;
    int phase = eventId & 0xff;
    uint32_t bounce = (uint32_t)(eventId >> 8);

    uint32_t width = ev->params.width, height = ev->params.height;
    MTLSize grid = MTLSizeMake(width, height, 1);
    MTLSize group = MTLSizeMake(8, 8, 1);

    auto barrier = ^(id<MTLComputeCommandEncoder> enc) {
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    };

    if (phase == kPhaseBegin)
    {
        if (ev->instanceCount <= 0 || ev->instanceCount > kMaxEventInstances)
            return;

        s_FrameTexture = (__bridge id<MTLTexture>)(void*)ev->texture;
        if (s_FrameTexture == nil ||
            !(s_FrameTexture.usage & MTLTextureUsageShaderWrite))
        {
            s_FrameTexture = nil;
            return;
        }

        bool resized = width != s_BufferWidth || height != s_BufferHeight;
        EnsurePathBuffers(width, height);

        s_Frame = {};
        s_Frame.cam = ev->params;
        std::memcpy(s_Frame.envColor, ev->envColor, sizeof(float) * 4);
        s_Frame.frameIndex = ev->frameIndex;
        s_Frame.maxBounces =
          std::min(std::max(ev->maxBounces, 1u), kMaxBounceLimit);
        s_Frame.linearOutput = ev->linearOutput;
        s_Frame.debugFlags = ev->debugFlags;
        s_Frame.exposure = ev->exposure;
        s_Frame.lightCount =
          std::min(ev->lightCount, (uint32_t)kMaxLights);
        std::memcpy(s_Frame.lights, ev->lights,
                    sizeof(LightDesc) * kMaxLights);

        id<MTLBuffer> info = nil;
        MTLInstanceAccelerationStructureDescriptor* descriptor =
          CreateInstanceDescriptor(ev->instances, ev->instanceCount, &info);
        if (descriptor == nil) return;

        id<MTLCommandBuffer> command = BeginPhaseCommandBuffer();
        if (command == nil) return;

        s_FrameTlas = EncodeAccelerationStructureBuild(command, descriptor);
        if (s_FrameTlas == nil) return;
        s_FrameInfo = info;

        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];

        if (ev->reset != 0 || resized)
        {
            s_AccumFrames = 0;
            [enc setComputePipelineState:s_ClearPipeline];
            [enc setBuffer:s_AccumBuffer offset:0 atIndex:0];
            [enc setBytes:&s_Frame length:sizeof(s_Frame) atIndex:1];
            [enc dispatchThreads:grid threadsPerThreadgroup:group];
            barrier(enc);
        }

        [enc setComputePipelineState:s_RayGenPipeline];
        [enc setBuffer:s_PathBuffer offset:0 atIndex:0];
        [enc setBytes:&s_Frame length:sizeof(s_Frame) atIndex:1];
        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];
        [command commit];
    }
    else if (phase == kPhaseIntersect)
    {
        if (s_FrameTlas == nil) return;

        id<MTLCommandBuffer> command = BeginPhaseCommandBuffer();
        if (command == nil) return;
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];

        UseSceneResources(enc);
        [enc useResource:s_FrameTlas usage:MTLResourceUsageRead];

        [enc setComputePipelineState:s_IntersectPipeline];
        [enc setAccelerationStructure:s_FrameTlas atBufferIndex:0];
        [enc setBuffer:s_PathBuffer offset:0 atIndex:1];
        [enc setBuffer:s_SharedHits offset:0 atIndex:2];
        [enc setBytes:&s_Frame length:sizeof(s_Frame) atIndex:3];
        [enc dispatchThreads:grid threadsPerThreadgroup:group];

        barrier(enc);

        [enc setComputePipelineState:s_GeomPrepPipeline];
        [enc setBuffer:s_PathBuffer offset:0 atIndex:0];
        [enc setBuffer:s_SharedHits offset:0 atIndex:1];
        [enc setBuffer:s_FrameInfo offset:0 atIndex:2];
        [enc setBuffer:s_MaterialBuffer offset:0 atIndex:3];
        [enc setBytes:&s_Frame length:sizeof(s_Frame) atIndex:4];
        [enc setBuffer:s_SharedAttributes offset:0 atIndex:5];
        [enc setBuffer:s_SharedSurfaces offset:0 atIndex:6];
        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];
        [command commit];
    }
    else if (phase == kPhaseShade)
    {
        if (s_FrameTlas == nil) return;

        id<MTLCommandBuffer> command = BeginPhaseCommandBuffer();
        if (command == nil) return;
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];

        UseSceneResources(enc);
        [enc useResource:s_FrameTlas usage:MTLResourceUsageRead];

        [enc setComputePipelineState:s_ShadePipeline];
        [enc setAccelerationStructure:s_FrameTlas atBufferIndex:0];
        [enc setBuffer:s_PathBuffer offset:0 atIndex:1];
        [enc setBuffer:s_SharedAttributes offset:0 atIndex:2];
        [enc setBuffer:s_SharedSurfaces offset:0 atIndex:3];
        [enc setBytes:&s_Frame length:sizeof(s_Frame) atIndex:4];
        [enc setBytes:&bounce length:sizeof(bounce) atIndex:5];
        [enc setBuffer:s_FrameInfo offset:0 atIndex:6];
        [enc setBuffer:s_MaterialBuffer offset:0 atIndex:7];
        [enc setBuffer:s_GBuffer offset:0 atIndex:8];
        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];
        [command commit];
    }
    else if (phase == kPhaseResolve)
    {
        if (s_FrameTexture == nil) return;

        id<MTLCommandBuffer> command = BeginPhaseCommandBuffer();
        if (command == nil) return;
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];

        [enc setComputePipelineState:s_ResolvePipeline];
        [enc setBuffer:s_PathBuffer offset:0 atIndex:0];
        [enc setBuffer:s_AccumBuffer offset:0 atIndex:1];
        [enc setBytes:&s_Frame length:sizeof(s_Frame) atIndex:2];
        [enc setBuffer:s_GBuffer offset:0 atIndex:3];
        [enc setBuffer:s_DenoiseA offset:0 atIndex:4];
        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        barrier(enc);

        // Edge-avoiding a-trous denoising during early accumulation only;
        // analytic (linear output) modes and converged frames pass through.
        int iterations = (s_Frame.linearOutput == 0 &&
                          s_AccumFrames < kDenoiseFrameLimit) ?
          kDenoiseIterations : 0;
        id<MTLBuffer> src = s_DenoiseA, dst = s_DenoiseB;
        for (int i = 0; i < iterations; i++)
        {
            uint32_t step = 1u << i;
            [enc setComputePipelineState:s_AtrousPipeline];
            [enc setBuffer:src offset:0 atIndex:0];
            [enc setBuffer:dst offset:0 atIndex:1];
            [enc setBuffer:s_GBuffer offset:0 atIndex:2];
            [enc setBytes:&s_Frame length:sizeof(s_Frame) atIndex:3];
            [enc setBytes:&step length:sizeof(step) atIndex:4];
            [enc dispatchThreads:grid threadsPerThreadgroup:group];
            barrier(enc);
            auto tmp = src; src = dst; dst = tmp;
        }

        [enc setComputePipelineState:s_OutputPipeline];
        [enc setBuffer:src offset:0 atIndex:0];
        [enc setBuffer:s_GBuffer offset:0 atIndex:1];
        [enc setBytes:&s_Frame length:sizeof(s_Frame) atIndex:2];
        [enc setTexture:s_FrameTexture atIndex:0];
        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];
        [command commit];

        s_AccumFrames++;
        s_EventFrames.fetch_add(1, std::memory_order_relaxed);
    }
}

} // anonymous namespace

extern "C" {

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
UnityPluginLoad(IUnityInterfaces* interfaces)
{
    s_Metal = interfaces->Get<IUnityGraphicsMetalV2>();
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
   uint32_t normalOffset, uint32_t tangentOffset, uint32_t uvOffset,
   uint32_t uv1Offset, uint32_t colorOffset, uint32_t colorFormat,
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
    mesh.tangentOffset = tangentOffset;
    mesh.uvOffset = uvOffset;
    mesh.uv1Offset = uv1Offset;
    mesh.colorOffset = colorOffset;
    mesh.colorFormat = colorFormat;
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
        mat.cutoff = src.cutoff;
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

// Registers the GraphicsBuffers shared with Unity compute shaders
// (hit records, hit attributes, surface records).
int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
MetalRT_SetSharedBuffers(void* hits, void* attributes, void* surfaces)
{
    if (hits == nullptr || attributes == nullptr || surfaces == nullptr)
    {
        SetError(@"Shared buffer pointer is null.");
        return -1;
    }
    s_SharedHits = (__bridge id<MTLBuffer>)hits;
    s_SharedAttributes = (__bridge id<MTLBuffer>)attributes;
    s_SharedSurfaces = (__bridge id<MTLBuffer>)surfaces;
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

UnityRenderingEventAndData UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
MetalRT_GetRenderEventFunc()
{
    return OnRenderEvent;
}

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
    s_SharedHits = nil;
    s_SharedAttributes = nil;
    s_SharedSurfaces = nil;
    s_PathBuffer = nil;
    s_AccumBuffer = nil;
    s_GBuffer = nil;
    s_DenoiseA = nil;
    s_DenoiseB = nil;
    s_FrameTlas = nil;
    s_FrameInfo = nil;
    s_FrameTexture = nil;
    s_BufferWidth = s_BufferHeight = 0;
    s_AccumFrames = 0;
    s_EventFrames.store(0, std::memory_order_relaxed);
}

} // extern "C"
