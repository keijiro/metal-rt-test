using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace MetalRTTest {

// P/Invoke interface to the MetalRTTest native plugin.
public static class MetalRTPlugin
{
    // Interop structs

    [StructLayout(LayoutKind.Sequential)]
    public struct TraceParams
    {
        public Vector4 originTan;   // xyz: ray origin (world space), w: tan(fovY/2)
        public Vector4 rightAspect; // xyz: camera right (world space), w: aspect
        public Vector4 up;          // xyz: camera up (world space)
        public Vector4 forward;     // xyz: camera forward (world space)
        public uint width, height, pad0, pad1;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ProbeResult
    {
        public float hit, distance, primitiveIndex, instanceIndex;
        public Vector2 barycentric, pad;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct InstanceDesc
    {
        public int meshIndex;
        public int materialIndex;
        public float pad0, pad1;
        public Vector4 objectToWorld0, objectToWorld1, objectToWorld2;
        public Vector4 normalMatrix0, normalMatrix1, normalMatrix2;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct MaterialDesc
    {
        public Vector4 baseColor; // linear
        public Vector4 emission;  // linear
        public Vector4 baseMapST; // xy: scale, zw: offset
        public float metallic, smoothness;
        public uint hasBaseMap, pad;
        public long texture;      // MTLTexture pointer or 0
        public long pad2;
    }

    // Per-frame settings carried in the event data blob.
    public struct FrameSettings
    {
        public Color envColor;    // linear radiance
        public Vector3 lightDir;  // direction the light travels
        public Color lightColor;  // linear color * intensity
        public bool reset;
        public uint maxBounces;
        public bool linearOutput; // skip tonemap/sRGB (analytic tests)
        public uint debugFlags;   // 1 = diffuse only
        public float exposure;
    }

    public const uint NoAttribute = 0xffffffffu;

    // Native entry points

    const string PluginName = "MetalRTTest";

    [DllImport(PluginName)] static extern IntPtr MetalRT_GetLastError();
    [DllImport(PluginName)] public static extern int
      MetalRT_DeviceSupportsRaytracing();
    [DllImport(PluginName)] public static extern int MetalRT_AddMesh
      (IntPtr vertexBuffer, uint vertexStride, uint positionOffset,
       uint normalOffset, uint uvOffset,
       IntPtr indexBuffer, uint indexFormat, uint indexByteOffset,
       uint triangleCount);
    [DllImport(PluginName)] public static extern int MetalRT_SetMaterials
      (MaterialDesc[] materials, int count);
    [DllImport(PluginName)] public static extern int MetalRT_BuildInstanceAS
      (InstanceDesc[] instances, int count);
    [DllImport(PluginName)] public static extern int MetalRT_TraceProbes
      (Vector4[] rays, int count, [Out] ProbeResult[] results);
    [DllImport(PluginName)] public static extern IntPtr
      MetalRT_GetRenderEventFunc();
    [DllImport(PluginName)] public static extern int
      MetalRT_GetEventFrameCount();
    [DllImport(PluginName)] public static extern void MetalRT_Dispose();

    public static string LastError
      => Marshal.PtrToStringAnsi(MetalRT_GetLastError());

    // Event data blob layout; must match EventData in MetalRTPlugin.mm.

    public const int MaxInstances = 16;

    static readonly int ParamsSize = Marshal.SizeOf<TraceParams>(); // 80
    static readonly int DescSize = Marshal.SizeOf<InstanceDesc>();  // 112
    const int HeaderSize = 176;

    public static int EventDataSize
      => HeaderSize + DescSize * MaxInstances;

    // Serializes one frame of trace input into an unmanaged event blob.
    public static void WriteEventData
      (IntPtr blob, in TraceParams p, IntPtr texture, uint frameIndex,
       in FrameSettings s, InstanceDesc[] instances)
    {
        Marshal.StructureToPtr(p, blob, false);
        Marshal.WriteInt64(blob, 80, texture.ToInt64());
        Marshal.WriteInt32(blob, 88, instances.Length);
        Marshal.WriteInt32(blob, 92, (int)frameIndex);
        WriteVec(blob, 96, s.envColor.r, s.envColor.g, s.envColor.b, 0);
        WriteVec(blob, 112, s.lightDir.x, s.lightDir.y, s.lightDir.z, 0);
        WriteVec(blob, 128, s.lightColor.r, s.lightColor.g, s.lightColor.b, 0);
        Marshal.WriteInt32(blob, 144, s.reset ? 1 : 0);
        Marshal.WriteInt32(blob, 148, (int)s.maxBounces);
        Marshal.WriteInt32(blob, 152, s.linearOutput ? 1 : 0);
        Marshal.WriteInt32(blob, 156, (int)s.debugFlags);
        WriteVec(blob, 160, s.exposure, 0, 0, 0);
        for (var i = 0; i < instances.Length; i++)
            Marshal.StructureToPtr
              (instances[i], IntPtr.Add(blob, HeaderSize + DescSize * i), false);
    }

    static void WriteVec(IntPtr blob, int offset,
                         float x, float y, float z, float w)
    {
        Marshal.WriteInt32(blob, offset, BitConverter.SingleToInt32Bits(x));
        Marshal.WriteInt32(blob, offset + 4, BitConverter.SingleToInt32Bits(y));
        Marshal.WriteInt32(blob, offset + 8, BitConverter.SingleToInt32Bits(z));
        Marshal.WriteInt32(blob, offset + 12, BitConverter.SingleToInt32Bits(w));
    }
}

} // namespace MetalRTTest
