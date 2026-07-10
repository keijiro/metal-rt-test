using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace UrpMetalPathTracer {

// P/Invoke interface to the MetalPathTracer native plugin.
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
        public float cutoff;      // alpha clip threshold; < 0 disables
        public uint hasBaseMap;
        public long texture;      // MTLTexture pointer or 0
        public long pad2;
    }

    // Punctual light description; must match LightDesc in MetalRTPlugin.mm.
    [StructLayout(LayoutKind.Sequential)]
    public struct LightDesc
    {
        public Vector4 position;  // xyz; w: type (0: dir, 1: point, 2: spot)
        public Vector4 direction; // xyz: direction the light travels; w: range
        public Vector4 color;     // linear color * intensity * pi
        public Vector4 spot;      // x: angle scale, y: angle offset
    }

    public const int MaxLights = 4;

    // Per-frame settings carried in the event data blob.
    public struct FrameSettings
    {
        public Color envColor;     // linear radiance
        public LightDesc[] lights; // up to MaxLights (null = none)
        public bool reset;
        public uint maxBounces;
        public bool linearOutput;  // skip tonemap/sRGB (analytic tests)
        public uint debugFlags;    // 1 = diffuse only
        public float exposure;
    }

    // Builds a LightDesc from a Unity Light (URP punctual conventions;
    // color premultiplied by pi for the normalized BRDF).
    public static LightDesc MakeLightDesc(Light light)
    {
        var type = light.type switch
        {
            LightType.Directional => 0f,
            LightType.Point => 1f,
            LightType.Spot => 2f,
            _ => -1f
        };
        var color = light.color.linear * (light.intensity * Mathf.PI);
        var desc = new LightDesc
        {
            position = light.transform.position,
            direction = light.transform.forward,
            color = new Vector4(color.r, color.g, color.b, 0)
        };
        desc.position.w = type;
        desc.direction.w = light.range;
        if (light.type == LightType.Spot)
        {
            var cosOuter = Mathf.Cos(light.spotAngle * Mathf.Deg2Rad / 2);
            var cosInner = Mathf.Cos(light.innerSpotAngle * Mathf.Deg2Rad / 2);
            var scale = 1 / Mathf.Max(cosInner - cosOuter, 1e-4f);
            desc.spot = new Vector4(scale, -cosOuter * scale, 0, 0);
        }
        return desc;
    }

    public const uint NoAttribute = 0xffffffffu;

    // Render event phases (eventId = phase | bounce << 8); must match
    // EventPhase in MetalRTPlugin.mm.
    public const int PhaseBegin = 0;
    public const int PhaseIntersect = 1;
    public const int PhaseShade = 2;
    public const int PhaseResolve = 3;

    // Shared buffer strides; must match the structs in MetalRTPlugin.mm and
    // TestProcedural.compute.
    public const int HitRecordStride = 32;
    public const int HitAttributesStride = 112;
    public const int SurfaceRecordStride = 64;

    // Native entry points

    const string PluginName = "MetalPathTracer";

    [DllImport(PluginName)] static extern IntPtr MetalRT_GetLastError();
    [DllImport(PluginName)] public static extern int
      MetalRT_DeviceSupportsRaytracing();
    [DllImport(PluginName)] public static extern int MetalRT_AddMesh
      (IntPtr vertexBuffer, uint vertexStride, uint positionOffset,
       uint normalOffset, uint tangentOffset, uint uvOffset,
       uint uv1Offset, uint colorOffset, uint colorFormat,
       IntPtr indexBuffer, uint indexFormat, uint indexByteOffset,
       uint triangleCount);
    [DllImport(PluginName)] public static extern int MetalRT_SetSharedBuffers
      (IntPtr hits, IntPtr attributes, IntPtr surfaces);
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
    static readonly int LightSize = Marshal.SizeOf<LightDesc>();    // 64
    const int LightsOffset = 144;
    static readonly int HeaderSize = LightsOffset + LightSize * MaxLights;

    public static int EventDataSize
      => HeaderSize + DescSize * MaxInstances;

    // Serializes one frame of trace input into an unmanaged event blob.
    public static void WriteEventData
      (IntPtr blob, in TraceParams p, IntPtr texture, uint frameIndex,
       in FrameSettings s, InstanceDesc[] instances)
    {
        var lightCount = Mathf.Min(s.lights?.Length ?? 0, MaxLights);

        Marshal.StructureToPtr(p, blob, false);
        Marshal.WriteInt64(blob, 80, texture.ToInt64());
        Marshal.WriteInt32(blob, 88, instances.Length);
        Marshal.WriteInt32(blob, 92, (int)frameIndex);
        WriteVec(blob, 96, s.envColor.r, s.envColor.g, s.envColor.b, 0);
        Marshal.WriteInt32(blob, 112, s.reset ? 1 : 0);
        Marshal.WriteInt32(blob, 116, (int)s.maxBounces);
        Marshal.WriteInt32(blob, 120, s.linearOutput ? 1 : 0);
        Marshal.WriteInt32(blob, 124, (int)s.debugFlags);
        Marshal.WriteInt32(blob, 128,
                           BitConverter.SingleToInt32Bits(s.exposure));
        Marshal.WriteInt32(blob, 132, lightCount);
        for (var i = 0; i < lightCount; i++)
            Marshal.StructureToPtr
              (s.lights[i], IntPtr.Add(blob, LightsOffset + LightSize * i),
               false);
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

} // namespace UrpMetalPathTracer
