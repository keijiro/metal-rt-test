// Metal hardware ray tracing test plugin for Unity
//
// Builds per-mesh primitive acceleration structures (BLAS) directly from
// Unity mesh GPU buffers (obtained via Mesh.GetNativeVertexBufferPtr /
// GetNativeIndexBufferPtr), combines them into an instance acceleration
// structure (TLAS) with per-instance transforms, and runs world-space
// intersection tests with metal::raytracing::intersector in compute kernels.
// Per-instance vertex/index buffers are accessed bindlessly through GPU
// addresses (Metal 3). All work runs synchronously on a private command
// queue created on Unity's MTLDevice, so no synchronization with Unity's
// render thread is needed.

#import <Metal/Metal.h>

#include "IUnityInterface.h"
#include "IUnityGraphicsMetal.h"

#include <cstring>
#include <string>
#include <vector>

namespace {

IUnityGraphicsMetalV1* s_Metal;

id<MTLDevice> s_Device;
id<MTLCommandQueue> s_Queue;
id<MTLComputePipelineState> s_TexturePipeline;
id<MTLComputePipelineState> s_ProbePipeline;

struct MeshEntry
{
    id<MTLAccelerationStructure> blas;
    id<MTLBuffer> vertexBuffer;
    id<MTLBuffer> indexBuffer;
    uint32_t vertexStride, positionOffset, indexFormat;
};

std::vector<MeshEntry> s_Meshes;
id<MTLAccelerationStructure> s_InstanceAS;
id<MTLBuffer> s_InstanceInfo;

std::string s_LastError;

// Must match TraceParams in MetalRayTracingTest.cs and in the shader source.
struct TraceParams
{
    float originTan[4];   // xyz: ray origin (world space), w: tan(fovY/2)
    float rightAspect[4]; // xyz: camera right (world space), w: aspect
    float up[4];          // xyz: camera up (world space)
    float forward[4];     // xyz: camera forward (world space)
    uint32_t width, height, pad0, pad1;
};

// Must match ProbeResult in MetalRayTracingTest.cs.
struct ProbeResult
{
    float hit, distance, primitiveIndex, instanceIndex;
    float barycentric[2], pad[2];
};

// Must match InstanceDesc in MetalRayTracingTest.cs.
struct InstanceDesc
{
    int32_t meshIndex;
    float pad[3];
    float objectToWorld[3][4]; // rows of the 3x4 object-to-world matrix
    float normalMatrix[3][4];  // rows (xyz used) of the world normal matrix
};

// Shader-side per-instance record; must match InstanceInfo in the shader
// source. Buffer pointers are stored as raw GPU addresses (Metal 3
// bindless), which the shader reads as device pointers.
struct GpuInstanceInfo
{
    uint64_t vertices, indices;
    uint32_t vertexStride, positionOffset, indexFormat, pad;
    float normalMatrix[3][4];
};

constexpr const char* kShaderSource = R"msl(
#include <metal_stdlib>
#include <metal_raytracing>

using namespace metal;
using namespace metal::raytracing;

struct TraceParams
{
    float4 originTan;
    float4 rightAspect;
    float4 up;
    float4 forward;
    uint width, height, pad0, pad1;
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
    uint vertexStride, positionOffset, indexFormat, pad;
    float4 normalMatrix0, normalMatrix1, normalMatrix2;
};

static ray GenerateRay(constant TraceParams& params, float2 ndc)
{
    float tanFov = params.originTan.w;
    float aspect = params.rightAspect.w;
    float3 dir = params.forward.xyz +
                 params.rightAspect.xyz * (ndc.x * tanFov * aspect) +
                 params.up.xyz * (ndc.y * tanFov);
    ray r;
    r.origin = params.originTan.xyz;
    r.direction = normalize(dir);
    r.min_distance = 0;
    r.max_distance = INFINITY;
    return r;
}

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

static float3 LoadPosition(constant InstanceInfo& info, uint index)
{
    device const packed_float3* p = (device const packed_float3*)
      (info.vertices + info.vertexStride * index + info.positionOffset);
    return *p;
}

// Flat-shaded world-space geometric normal of the hit triangle, oriented
// toward the ray origin.
static float3 HitNormal
  (constant InstanceInfo& info, uint prim, float3 rayDirection)
{
    uint3 tri = LoadTriangleIndices(info, prim);
    float3 p0 = LoadPosition(info, tri.x);
    float3 p1 = LoadPosition(info, tri.y);
    float3 p2 = LoadPosition(info, tri.z);
    float3 n = cross(p1 - p0, p2 - p0); // object space
    n = float3(dot(info.normalMatrix0.xyz, n),
               dot(info.normalMatrix1.xyz, n),
               dot(info.normalMatrix2.xyz, n));
    n = normalize(n);
    return dot(n, rayDirection) > 0 ? -n : n;
}

kernel void TraceImageTexture
  (instance_acceleration_structure accel [[buffer(0)]],
   constant TraceParams& params [[buffer(1)]],
   constant InstanceInfo* instances [[buffer(2)]],
   texture2d<float, access::write> output [[texture(0)]],
   uint2 id [[thread_position_in_grid]])
{
    if (id.x >= params.width || id.y >= params.height) return;

    float2 ndc = float2(id) / float2(params.width, params.height) * 2 - 1;
    ray r = GenerateRay(params, ndc);

    intersector<triangle_data, instancing> isect;
    intersection_result<triangle_data, instancing> hit =
      isect.intersect(r, accel, 0xffu);

    float3 color = float3(0.1, 0.1, 0.2); // background

    if (hit.type == intersection_type::triangle)
    {
        constant InstanceInfo& info = instances[hit.instance_id];
        float3 n = HitNormal(info, hit.primitive_id, r.direction);
        color = n * 0.5 + 0.5;
    }

    output.write(float4(saturate(color), 1), id);
}

// Probe rays are given explicitly in world space as (origin, direction)
// float4 pairs, so results can be checked against analytically derived
// expectations on the CPU side.
kernel void TraceProbes
  (instance_acceleration_structure accel [[buffer(0)]],
   constant TraceParams& params [[buffer(1)]],
   device ProbeResult* output [[buffer(2)]],
   constant InstanceInfo* instances [[buffer(3)]],
   constant float4* rays [[buffer(4)]],
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
    if (s_TexturePipeline != nil && s_ProbePipeline != nil) return true;

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

    s_TexturePipeline = CreatePipeline(library, @"TraceImageTexture");
    s_ProbePipeline = CreatePipeline(library, @"TraceProbes");
    return s_TexturePipeline != nil && s_ProbePipeline != nil;
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

id<MTLAccelerationStructure> BuildAccelerationStructure
  (MTLAccelerationStructureDescriptor* descriptor)
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

    id<MTLCommandBuffer> command = [s_Queue commandBuffer];
    id<MTLAccelerationStructureCommandEncoder> encoder =
      [command accelerationStructureCommandEncoder];
    [encoder buildAccelerationStructure:accel
                             descriptor:descriptor
                          scratchBuffer:scratch
                    scratchBufferOffset:0];
    [encoder endEncoding];

    return RunCommandBuffer(command) ? accel : nil;
}

// Marks all indirectly referenced resources (BLASes and bindlessly accessed
// mesh buffers) resident for the dispatch.
void UseMeshResources(id<MTLComputeCommandEncoder> encoder)
{
    for (const auto& mesh : s_Meshes)
    {
        [encoder useResource:mesh.blas usage:MTLResourceUsageRead];
        [encoder useResource:mesh.vertexBuffer usage:MTLResourceUsageRead];
        [encoder useResource:mesh.indexBuffer usage:MTLResourceUsageRead];
    }
}

bool DispatchTrace(id<MTLComputePipelineState> pipeline,
                   const TraceParams* params,
                   void (^bind)(id<MTLComputeCommandEncoder>),
                   MTLSize grid, MTLSize group)
{
    id<MTLCommandBuffer> command = [s_Queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];
    [encoder setAccelerationStructure:s_InstanceAS atBufferIndex:0];
    [encoder setBytes:params length:sizeof(TraceParams) atIndex:1];
    UseMeshResources(encoder);
    bind(encoder);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    [encoder endEncoding];

    return RunCommandBuffer(command);
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

// Registers a mesh and builds its BLAS. Returns the mesh index (>= 0) on
// success or a negative error code.
int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
MetalRT_AddMesh
  (void* vertexBuffer, uint32_t vertexStride, uint32_t positionOffset,
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

    mesh.blas = BuildAccelerationStructure(descriptor);
    if (mesh.blas == nil) return -3;

    s_Meshes.push_back(mesh);
    return (int)s_Meshes.size() - 1;
}

// (Re)builds the instance acceleration structure (TLAS) and the shader-side
// per-instance records. Cheap enough to call every frame with updated
// transforms.
int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
MetalRT_BuildInstanceAS(const InstanceDesc* instances, int32_t count)
{
    if (!EnsureDevice()) return -1;

    if (s_Meshes.empty() || count <= 0)
    {
        SetError(@"No meshes registered or empty instance list.");
        return -2;
    }

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
            return -3;
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
        info.indexFormat = mesh.indexFormat;
        info.pad = 0;
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

    s_InstanceAS = BuildAccelerationStructure(descriptor);
    if (s_InstanceAS == nil) return -4;

    s_InstanceInfo = infoBuffer;
    return 0;
}

// Writes directly into a Unity RenderTexture (created with
// enableRandomWrite) given its GetNativeTexturePtr. The dispatch completes
// synchronously, so Unity can sample the texture afterwards without any
// extra synchronization.
int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
MetalRT_TraceToTexture(const TraceParams* params, void* texturePtr)
{
    if (s_InstanceAS == nil)
    {
        SetError(@"Instance acceleration structure has not been built.");
        return -1;
    }
    if (!EnsurePipelines()) return -2;

    id<MTLTexture> texture = (__bridge id<MTLTexture>)texturePtr;
    if (texture == nil || !(texture.usage & MTLTextureUsageShaderWrite))
    {
        SetError(@"Target texture is nil or lacks shader write usage.");
        return -3;
    }

    bool ok = DispatchTrace(
      s_TexturePipeline, params,
      ^(id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:s_InstanceInfo offset:0 atIndex:2];
          [encoder setTexture:texture atIndex:0];
      },
      MTLSizeMake(params->width, params->height, 1), MTLSizeMake(8, 8, 1));
    return ok ? 0 : -4;
}

// rays: (origin.xyzw, direction.xyzw) float4 pairs in world space
int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
MetalRT_TraceProbes(const TraceParams* params,
                    const float* rays, int32_t count, ProbeResult* results)
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

    bool ok = DispatchTrace(
      s_ProbePipeline, params,
      ^(id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:output offset:0 atIndex:2];
          [encoder setBuffer:s_InstanceInfo offset:0 atIndex:3];
          [encoder setBytes:rays length:sizeof(float) * 8 * count atIndex:4];
      },
      MTLSizeMake(count, 1, 1), MTLSizeMake(1, 1, 1));
    if (!ok) return -3;

    std::memcpy(results, output.contents, size);
    return 0;
}

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API MetalRT_Dispose()
{
    s_InstanceAS = nil;
    s_InstanceInfo = nil;
    s_Meshes.clear();
}

} // extern "C"
