using System;
using System.IO;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

namespace MetalRTTest {

// Verifies that Metal acceleration structures can be built from non-readable
// Unity meshes (GPU buffers only) and that hardware ray intersection works,
// via the MetalRTTest native plugin. Per-mesh BLASes are built once, then an
// instance acceleration structure (TLAS) is rebuilt every frame from the
// scene transforms and traced in world space. Shows a rasterized reference
// (left) and the ray traced result (right) side by side.
public sealed class MetalRayTracingTest : MonoBehaviour
{
    // Native plugin interface

    [StructLayout(LayoutKind.Sequential)]
    struct TraceParams
    {
        public Vector4 originTan;   // xyz: ray origin (world space), w: tan(fovY/2)
        public Vector4 rightAspect; // xyz: camera right (world space), w: aspect
        public Vector4 up;          // xyz: camera up (world space)
        public Vector4 forward;     // xyz: camera forward (world space)
        public uint width, height, pad0, pad1;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct ProbeResult
    {
        public float hit, distance, primitiveIndex, instanceIndex;
        public Vector2 barycentric, pad;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct InstanceDesc
    {
        public int meshIndex;
        public float pad0, pad1, pad2;
        public Vector4 objectToWorld0, objectToWorld1, objectToWorld2;
        public Vector4 normalMatrix0, normalMatrix1, normalMatrix2;
    }

    const string PluginName = "MetalRTTest";

    [DllImport(PluginName)] static extern IntPtr MetalRT_GetLastError();
    [DllImport(PluginName)] static extern int MetalRT_DeviceSupportsRaytracing();
    [DllImport(PluginName)] static extern int MetalRT_AddMesh
      (IntPtr vertexBuffer, uint vertexStride, uint positionOffset,
       IntPtr indexBuffer, uint indexFormat, uint indexByteOffset,
       uint triangleCount);
    [DllImport(PluginName)] static extern int MetalRT_BuildInstanceAS
      (InstanceDesc[] instances, int count);
    [DllImport(PluginName)] static extern int MetalRT_TraceProbes
      (in TraceParams traceParams, Vector4[] rays, int count,
       [Out] ProbeResult[] results);
    [DllImport(PluginName)] static extern IntPtr MetalRT_GetRenderEventFunc();
    [DllImport(PluginName)] static extern int MetalRT_GetEventFrameCount();
    [DllImport(PluginName)] static extern void MetalRT_Dispose();

    static string LastError
      => Marshal.PtrToStringAnsi(MetalRT_GetLastError());

    // Scene bootstrap

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Bootstrap()
      => new GameObject("Metal RT Test", typeof(MetalRayTracingTest));

    // World-space probe rays with analytic expectations for the initial
    // (unrotated) scene layout: torus (R=1.0, r=0.4) at the origin, sphere
    // (radius 0.5, scale 1.2) at (2.2, 0.6, 0.5).
    static readonly (Vector3 origin, Vector3 dir,
                     bool hit, float dist, int instance)[] Probes =
    {
        // inner tube wall of the origin torus
        (Vector3.zero, Vector3.right, true, 0.6f, 0),
        // through the torus hole (sphere is out of this path)
        (new Vector3(0, 5, 0), Vector3.down, false, 0, 0),
        // top of the sphere (center y 0.6 + radius 0.6)
        (new Vector3(2.2f, 5, 0.5f), Vector3.down, true, 3.8f, 1),
        // pointing away from everything
        (new Vector3(0, 0, -5), Vector3.back, false, 0, 0),
        // outer wall of the origin torus (small torus is out of reach)
        (new Vector3(-5, 0, 0), Vector3.right, true, 3.6f, 0),
    };

    const float ProbeTolerance = 0.02f;

    // Private members

    Camera _camera;
    (int mesh, Transform transform)[] _instances;
    RenderTexture _result;
    IntPtr _resultPtr;
    CommandBuffer _traceCommands;
    IntPtr _eventFunc;
    IntPtr[] _eventData;
    int _eventSlot;
    bool _traceLogged;
    int _tracedFrames;

    // Event data blob layout; must match EventData in MetalRTPlugin.mm.
    const int MaxInstances = 16;
    const int EventRingSize = 4;
    static readonly int ParamsSize = Marshal.SizeOf<TraceParams>();
    static readonly int DescSize = Marshal.SizeOf<InstanceDesc>();
    static int EventDataSize => ParamsSize + 16 + DescSize * MaxInstances;

    static void Log(string message) => Debug.Log("[MetalRT] " + message);

    // MonoBehaviour implementation

    void Start()
    {
        Log($"Graphics device: {SystemInfo.graphicsDeviceType} " +
            $"({SystemInfo.graphicsDeviceName})");
        Log($"Unity API: supportsRayTracing = {SystemInfo.supportsRayTracing}, " +
            $"supportsInlineRayTracing = {SystemInfo.supportsInlineRayTracing}");
        Log($"Native Metal device supportsRaytracing = " +
            $"{MetalRT_DeviceSupportsRaytracing()} (expected 1)");

        if (!SetUpScene()) return;
        if (!BuildInstanceAS()) return;
        RunProbeTest();
        SetUpRenderTarget();
    }

    void Update()
    {
        if (_result == null) return;

        // Animate the tori so the per-frame TLAS rebuild visibly stays in
        // sync with the rasterized reference.
        _instances[0].transform.Rotate(0, 20 * Time.deltaTime, 0, Space.World);
        _instances[2].transform.Rotate(15 * Time.deltaTime, 0, 0, Space.Self);

        TraceFrame();

        if (++_tracedFrames == 10)
        {
            Log($"Render thread events executed: " +
                $"{MetalRT_GetEventFrameCount()} (traced frames: " +
                $"{_tracedFrames}, expected roughly equal)");
            SaveResultImage();
        }
    }

    void OnDestroy()
    {
        MetalRT_Dispose();
        _traceCommands?.Dispose();
        if (_eventData != null)
            foreach (var ptr in _eventData) Marshal.FreeHGlobal(ptr);
    }

    void OnGUI()
    {
        if (_result == null) return;
        var w = Screen.width / 2;
        GUI.DrawTexture(new Rect(w, 0, w, Screen.height), _result,
                        ScaleMode.StretchToFill);
        GUI.Label(new Rect(10, 10, 300, 20), "Raster (reference)");
        GUI.Label(new Rect(w + 10, 10, 300, 20), "Metal RT");
    }

    // Test scene construction

    bool SetUpScene()
    {
        var torus = LoadAndRegisterMesh("Torus");
        var sphere = LoadAndRegisterMesh("Sphere");
        if (torus < 0 || sphere < 0) return false;

        // Initial transforms must match the probe expectations above.
        _instances = new[]
        {
            (torus, SpawnInstance("Torus", torus,
              Vector3.zero, Quaternion.identity, 1)),
            (sphere, SpawnInstance("Sphere", sphere,
              new Vector3(2.2f, 0.6f, 0.5f), Quaternion.identity, 1.2f)),
            (torus, SpawnInstance("Small Torus", torus,
              new Vector3(-2.1f, 0.3f, 0.8f), Quaternion.Euler(70, 0, 20), 0.5f)),
        };

        _camera = Camera.main;
        if (_camera == null)
            _camera = new GameObject("Camera", typeof(Camera))
                        .GetComponent<Camera>();
        _camera.transform.position = new Vector3(2.8f, 3.2f, -5.8f);
        _camera.transform.LookAt(new Vector3(0.15f, 0.1f, 0));
        _camera.fieldOfView = 45;
        _camera.rect = new Rect(0, 0, 0.5f, 1); // left half: raster reference
        _camera.clearFlags = CameraClearFlags.SolidColor;
        _camera.backgroundColor = new Color(0.1f, 0.1f, 0.2f);
        return true;
    }

    Transform SpawnInstance(string name, int meshIndex,
                            Vector3 position, Quaternion rotation, float scale)
    {
        var mesh = _meshes[meshIndex];
        var go = new GameObject(name, typeof(MeshFilter), typeof(MeshRenderer));
        go.transform.SetPositionAndRotation(position, rotation);
        go.transform.localScale = Vector3.one * scale;
        go.GetComponent<MeshFilter>().sharedMesh = mesh;
        go.GetComponent<MeshRenderer>().sharedMaterial =
          _material ??= new Material(Shader.Find("MetalRT/WorldNormal"));
        return go.transform;
    }

    Mesh[] _meshes = new Mesh[0];
    Material _material;

    // BLAS construction from a non-readable mesh asset

    int LoadAndRegisterMesh(string resourceName)
    {
        var mesh = Resources.Load<Mesh>(resourceName);
        Log($"Mesh {resourceName}: {mesh.vertexCount} verts, isReadable = " +
            $"{mesh.isReadable} (expected False)");

        var posStream = mesh.GetVertexAttributeStream(VertexAttribute.Position);
        var posOffset = mesh.GetVertexAttributeOffset(VertexAttribute.Position);
        var posFormat = mesh.GetVertexAttributeFormat(VertexAttribute.Position);
        var posDim = mesh.GetVertexAttributeDimension(VertexAttribute.Position);
        var stride = mesh.GetVertexBufferStride(posStream);

        if (posFormat != VertexAttributeFormat.Float32 || posDim != 3 ||
            mesh.GetTopology(0) != MeshTopology.Triangles ||
            mesh.GetBaseVertex(0) != 0)
        {
            Log($"FAIL: Unsupported mesh layout in {resourceName}");
            return -1;
        }

        var indexCount = mesh.GetIndexCount(0);
        var indexStart = mesh.GetIndexStart(0);
        var is16Bit = mesh.indexFormat == IndexFormat.UInt16;
        var indexSize = is16Bit ? 2u : 4u;

        // Native buffer pointers are only guaranteed valid within this
        // frame, so the BLAS is built right away (synchronously).
        var ret = MetalRT_AddMesh
          (mesh.GetNativeVertexBufferPtr(posStream), (uint)stride,
           (uint)posOffset, mesh.GetNativeIndexBufferPtr(),
           is16Bit ? 0u : 1u, indexStart * indexSize, indexCount / 3);

        if (ret < 0)
        {
            Log($"FAIL: BLAS build error {ret} for {resourceName}: {LastError}");
            return -1;
        }

        Log($"BLAS #{ret} built from non-readable mesh {resourceName} " +
            $"({indexCount / 3} triangles): OK");

        Array.Resize(ref _meshes, ret + 1);
        _meshes[ret] = mesh;
        return ret;
    }

    // TLAS instance descriptors from the current scene transforms

    InstanceDesc MakeInstanceDesc(int index)
    {
        var (meshIndex, transform) = _instances[index];
        var l2w = transform.localToWorldMatrix;
        var nrm = l2w.inverse.transpose;
        return new InstanceDesc
        {
            meshIndex = meshIndex,
            objectToWorld0 = l2w.GetRow(0),
            objectToWorld1 = l2w.GetRow(1),
            objectToWorld2 = l2w.GetRow(2),
            normalMatrix0 = nrm.GetRow(0),
            normalMatrix1 = nrm.GetRow(1),
            normalMatrix2 = nrm.GetRow(2)
        };
    }

    // Synchronous TLAS build used by the probe test path.
    bool BuildInstanceAS()
    {
        var descs = new InstanceDesc[_instances.Length];
        for (var i = 0; i < _instances.Length; i++)
            descs[i] = MakeInstanceDesc(i);

        var ret = MetalRT_BuildInstanceAS(descs, descs.Length);
        if (ret != 0)
        {
            Log($"FAIL: Instance AS build error {ret}: {LastError}");
            return false;
        }

        Log($"Instance AS (TLAS) built with {descs.Length} instances " +
            $"over {_meshes.Length} BLASes: OK");
        return true;
    }

    // Data level test: analytically verifiable world-space probe rays

    void RunProbeTest()
    {
        var rays = new Vector4[Probes.Length * 2];
        for (var i = 0; i < Probes.Length; i++)
        {
            rays[i * 2] = Probes[i].origin;
            rays[i * 2 + 1] = Probes[i].dir;
        }

        var results = new ProbeResult[Probes.Length];
        var ret = MetalRT_TraceProbes(default, rays, Probes.Length, results);
        if (ret != 0)
        {
            Log($"FAIL: Probe trace error {ret}: {LastError}");
            return;
        }

        var passed = 0;
        for (var i = 0; i < Probes.Length; i++)
        {
            var (origin, dir, expHit, expDist, expInst) = Probes[i];
            var r = results[i];
            var hit = r.hit > 0.5f;
            var ok = hit == expHit &&
                     (!expHit ||
                      (Mathf.Abs(r.distance - expDist) < ProbeTolerance &&
                       (int)r.instanceIndex == expInst));
            if (ok) passed++;

            var expected = expHit ?
              $"hit at t={expDist} on instance {expInst}" : "miss";
            var actual = hit ?
              $"hit at t={r.distance:F4} on instance {(int)r.instanceIndex} " +
              $"(tri {r.primitiveIndex})" : "miss";
            Log($"Probe {i}: {origin} -> {dir}: {actual}; " +
                $"expected {expected} ... {(ok ? "PASS" : "FAIL")}");
        }

        Log($"Probe test: {passed}/{Probes.Length} passed" +
            (passed == Probes.Length ? " -- ALL PASS" : " -- FAILURE"));
    }

    // Visual test: per-frame TLAS rebuild and full trace, encoded into
    // Unity's own Metal command stream on the render thread via
    // CommandBuffer.IssuePluginEventAndData (no CPU stall).

    void SetUpRenderTarget()
    {
        var (w, h) = (_camera.pixelWidth, _camera.pixelHeight);
        _result = new RenderTexture(w, h, 0, RenderTextureFormat.ARGB32)
          { enableRandomWrite = true };
        _result.Create();
        _resultPtr = _result.GetNativeTexturePtr();
        Log($"RenderTexture ({w}x{h}) created; native ptr acquired");

        _eventFunc = MetalRT_GetRenderEventFunc();
        _traceCommands = new CommandBuffer { name = "MetalRT Trace" };

        // A small ring of unmanaged blobs so the render thread never reads
        // a blob the main thread is currently rewriting.
        _eventData = new IntPtr[EventRingSize];
        for (var i = 0; i < EventRingSize; i++)
            _eventData[i] = Marshal.AllocHGlobal(EventDataSize);
    }

    void TraceFrame()
    {
        var ct = _camera.transform;
        var tanFov = Mathf.Tan(_camera.fieldOfView * Mathf.Deg2Rad / 2);
        var (w, h) = (_result.width, _result.height);

        var p = new TraceParams
        {
            originTan = V4(ct.position, tanFov),
            rightAspect = V4(ct.right, (float)w / h),
            up = V4(ct.up, 0),
            forward = V4(ct.forward, 0),
            width = (uint)w,
            height = (uint)h
        };

        _traceCommands.Clear();
        _traceCommands.IssuePluginEventAndData(_eventFunc, 0, WriteEventData(p));
        Graphics.ExecuteCommandBuffer(_traceCommands);

        if (!_traceLogged)
        {
            _traceLogged = true;
            Log("Render thread trace path active " +
                "(TLAS rebuild + trace on Unity's command stream)");
        }
    }

    // Serializes this frame's trace parameters and instance descriptors into
    // the next unmanaged blob of the ring.
    IntPtr WriteEventData(in TraceParams p)
    {
        _eventSlot = (_eventSlot + 1) % EventRingSize;
        var ptr = _eventData[_eventSlot];
        Marshal.StructureToPtr(p, ptr, false);
        Marshal.WriteInt64(ptr, ParamsSize, _resultPtr.ToInt64());
        Marshal.WriteInt32(ptr, ParamsSize + 8, _instances.Length);
        Marshal.WriteInt32(ptr, ParamsSize + 12, 0);
        for (var i = 0; i < _instances.Length; i++)
            Marshal.StructureToPtr
              (MakeInstanceDesc(i),
               IntPtr.Add(ptr, ParamsSize + 16 + DescSize * i), false);
        return ptr;
    }

    // Reads the natively written RenderTexture back on the Unity side, which
    // also verifies the texture contents are visible to Unity.
    void SaveResultImage()
    {
        var (w, h) = (_result.width, _result.height);
        var prev = RenderTexture.active;
        RenderTexture.active = _result;
        var tex = new Texture2D(w, h, TextureFormat.RGBA32, false);
        tex.ReadPixels(new Rect(0, 0, w, h), 0, 0);
        tex.Apply();
        RenderTexture.active = prev;

        var path = Path.GetFullPath
          (Path.Combine(Application.dataPath, "..", "Output", "rt-result.png"));
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllBytes(path, tex.EncodeToPNG());
        Destroy(tex);
        Log($"Ray traced image ({w}x{h}) written to {path}");
    }

    static Vector4 V4(Vector3 v, float w) => new Vector4(v.x, v.y, v.z, w);
}

} // namespace MetalRTTest
