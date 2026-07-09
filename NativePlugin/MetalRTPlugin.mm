// Metal hardware ray tracing test plugin for Unity
//
// Builds a MTLAccelerationStructure directly from Unity mesh GPU buffers
// (obtained via Mesh.GetNativeVertexBufferPtr / GetNativeIndexBufferPtr) and
// runs intersection tests with metal::raytracing::intersector in a compute
// kernel. All work runs synchronously on a private command queue created on
// Unity's MTLDevice, so no synchronization with Unity's render thread is
// needed.

#import <Metal/Metal.h>

#include "IUnityInterface.h"
#include "IUnityGraphicsMetal.h"

#include <cstring>
#include <string>

namespace {

IUnityGraphicsMetalV1* s_Metal;

id<MTLDevice> s_Device;
id<MTLCommandQueue> s_Queue;
id<MTLComputePipelineState> s_ImagePipeline;
id<MTLComputePipelineState> s_ProbePipeline;

id<MTLAccelerationStructure> s_Accel;
id<MTLBuffer> s_VertexBuffer;
id<MTLBuffer> s_IndexBuffer;

std::string s_LastError;

// Must match TraceParams in MetalRayTracingTest.cs and in the shader source.
struct TraceParams
{
    float originTan[4];   // xyz: ray origin (object space), w: tan(fovY/2)
    float rightAspect[4]; // xyz: camera right (object space), w: aspect
    float up[4];          // xyz: camera up (object space)
    float forward[4];     // xyz: camera forward (object space)
    uint32_t width, height, vertexStride, posOffset;
    uint32_t indexFormat; // 0: UInt16, 1: UInt32
    uint32_t pad[3];
};

// Must match ProbeResult in MetalRayTracingTest.cs.
struct ProbeResult
{
    float hit, distance, primitiveIndex, pad;
    float barycentric[2], pad2[2];
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
    uint width, height, vertexStride, posOffset;
    uint indexFormat;
    uint pad[3];
};

struct ProbeResult
{
    float hit, distance, primitiveIndex, pad;
    float2 barycentric, pad2;
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

static uint3 LoadTriangleIndices
  (constant TraceParams& params, device const uchar* indices, uint prim)
{
    if (params.indexFormat == 0)
    {
        device const ushort* p = (device const ushort*)indices;
        return uint3(p[prim * 3], p[prim * 3 + 1], p[prim * 3 + 2]);
    }
    else
    {
        device const uint* p = (device const uint*)indices;
        return uint3(p[prim * 3], p[prim * 3 + 1], p[prim * 3 + 2]);
    }
}

static float3 LoadPosition
  (constant TraceParams& params, device const uchar* vertices, uint index)
{
    device const packed_float3* p = (device const packed_float3*)
      (vertices + params.vertexStride * index + params.posOffset);
    return *p;
}

kernel void TraceImage
  (primitive_acceleration_structure accel [[buffer(0)]],
   constant TraceParams& params [[buffer(1)]],
   device uchar4* output [[buffer(2)]],
   device const uchar* vertices [[buffer(3)]],
   device const uchar* indices [[buffer(4)]],
   uint2 id [[thread_position_in_grid]])
{
    if (id.x >= params.width || id.y >= params.height) return;

    float2 ndc = float2(id) / float2(params.width, params.height) * 2 - 1;
    ray r = GenerateRay(params, ndc);

    intersector<triangle_data> isect;
    intersection_result<triangle_data> hit = isect.intersect(r, accel);

    float3 color = float3(0.1, 0.1, 0.2); // background

    if (hit.type == intersection_type::triangle)
    {
        uint3 tri = LoadTriangleIndices(params, indices, hit.primitive_id);
        float3 p0 = LoadPosition(params, vertices, tri.x);
        float3 p1 = LoadPosition(params, vertices, tri.y);
        float3 p2 = LoadPosition(params, vertices, tri.z);
        float3 n = normalize(cross(p1 - p0, p2 - p0));
        if (dot(n, r.direction) > 0) n = -n; // face the ray origin
        color = n * 0.5 + 0.5;
    }

    output[id.y * params.width + id.x] =
      uchar4(uchar3(saturate(color) * 255), 255);
}

// Probe rays are given explicitly in object space as (origin, direction)
// float4 pairs, so results can be checked against analytically derived
// expectations on the CPU side.
kernel void TraceProbes
  (primitive_acceleration_structure accel [[buffer(0)]],
   constant TraceParams& params [[buffer(1)]],
   device ProbeResult* output [[buffer(2)]],
   constant float4* rays [[buffer(3)]],
   uint id [[thread_position_in_grid]])
{
    ray r;
    r.origin = rays[id * 2].xyz;
    r.direction = normalize(rays[id * 2 + 1].xyz);
    r.min_distance = 0;
    r.max_distance = INFINITY;

    intersector<triangle_data> isect;
    intersection_result<triangle_data> hit = isect.intersect(r, accel);

    ProbeResult res = {};
    if (hit.type == intersection_type::triangle)
    {
        res.hit = 1;
        res.distance = hit.distance;
        res.primitiveIndex = hit.primitive_id;
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
    if (s_ImagePipeline != nil && s_ProbePipeline != nil) return true;

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

    s_ImagePipeline = CreatePipeline(library, @"TraceImage");
    s_ProbePipeline = CreatePipeline(library, @"TraceProbes");
    return s_ImagePipeline != nil && s_ProbePipeline != nil;
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

bool DispatchTrace(id<MTLComputePipelineState> pipeline,
                   const TraceParams* params,
                   id<MTLBuffer> outputBuffer,
                   void (^bindExtra)(id<MTLComputeCommandEncoder>),
                   MTLSize grid, MTLSize group)
{
    id<MTLCommandBuffer> command = [s_Queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];
    [encoder setAccelerationStructure:s_Accel atBufferIndex:0];
    [encoder setBytes:params length:sizeof(TraceParams) atIndex:1];
    [encoder setBuffer:outputBuffer offset:0 atIndex:2];
    bindExtra(encoder);
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

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
MetalRT_BuildAccelerationStructure
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

    s_VertexBuffer = (__bridge id<MTLBuffer>)vertexBuffer;
    s_IndexBuffer = (__bridge id<MTLBuffer>)indexBuffer;

    MTLAccelerationStructureTriangleGeometryDescriptor* geometry =
      [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    geometry.vertexBuffer = s_VertexBuffer;
    geometry.vertexBufferOffset = positionOffset;
    geometry.vertexStride = vertexStride;
    geometry.vertexFormat = MTLAttributeFormatFloat3;
    geometry.indexBuffer = s_IndexBuffer;
    geometry.indexBufferOffset = indexByteOffset;
    geometry.indexType =
      indexFormat == 0 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
    geometry.triangleCount = triangleCount;
    geometry.opaque = YES;

    MTLPrimitiveAccelerationStructureDescriptor* descriptor =
      [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    descriptor.geometryDescriptors = @[geometry];

    MTLAccelerationStructureSizes sizes =
      [s_Device accelerationStructureSizesWithDescriptor:descriptor];

    s_Accel =
      [s_Device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
    id<MTLBuffer> scratch =
      [s_Device newBufferWithLength:sizes.buildScratchBufferSize
                            options:MTLResourceStorageModePrivate];
    if (s_Accel == nil || scratch == nil)
    {
        SetError(@"Acceleration structure allocation failed.");
        return -3;
    }

    id<MTLCommandBuffer> command = [s_Queue commandBuffer];
    id<MTLAccelerationStructureCommandEncoder> encoder =
      [command accelerationStructureCommandEncoder];
    [encoder buildAccelerationStructure:s_Accel
                             descriptor:descriptor
                          scratchBuffer:scratch
                    scratchBufferOffset:0];
    [encoder endEncoding];

    return RunCommandBuffer(command) ? 0 : -4;
}

int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
MetalRT_TraceImage(const TraceParams* params, void* outputPixels)
{
    if (s_Accel == nil)
    {
        SetError(@"Acceleration structure has not been built.");
        return -1;
    }
    if (!EnsurePipelines()) return -2;

    NSUInteger size = params->width * params->height * 4;
    id<MTLBuffer> output =
      [s_Device newBufferWithLength:size options:MTLResourceStorageModeShared];

    bool ok = DispatchTrace(
      s_ImagePipeline, params, output,
      ^(id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:s_VertexBuffer offset:0 atIndex:3];
          [encoder setBuffer:s_IndexBuffer offset:0 atIndex:4];
      },
      MTLSizeMake(params->width, params->height, 1), MTLSizeMake(8, 8, 1));
    if (!ok) return -3;

    std::memcpy(outputPixels, output.contents, size);
    return 0;
}

// rays: (origin.xyzw, direction.xyzw) float4 pairs in object space
int UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
MetalRT_TraceProbes(const TraceParams* params,
                    const float* rays, int32_t count, ProbeResult* results)
{
    if (s_Accel == nil)
    {
        SetError(@"Acceleration structure has not been built.");
        return -1;
    }
    if (!EnsurePipelines()) return -2;

    NSUInteger size = sizeof(ProbeResult) * count;
    id<MTLBuffer> output =
      [s_Device newBufferWithLength:size options:MTLResourceStorageModeShared];

    bool ok = DispatchTrace(
      s_ProbePipeline, params, output,
      ^(id<MTLComputeCommandEncoder> encoder) {
          [encoder setBytes:rays length:sizeof(float) * 8 * count atIndex:3];
      },
      MTLSizeMake(count, 1, 1), MTLSizeMake(1, 1, 1));
    if (!ok) return -3;

    std::memcpy(results, output.contents, size);
    return 0;
}

void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API MetalRT_Dispose()
{
    s_Accel = nil;
    s_VertexBuffer = nil;
    s_IndexBuffer = nil;
}

} // extern "C"
