using System;
using System.IO;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

namespace MetalRTTest {

// Verifies that a Metal acceleration structure can be built from a
// non-readable Unity mesh (GPU buffers only) and that hardware ray
// intersection works, via the MetalRTTest native plugin. Shows a rasterized
// reference (left) and the ray traced result (right) side by side.
public sealed class MetalRayTracingTest : MonoBehaviour
{
    // Native plugin interface

    [StructLayout(LayoutKind.Sequential)]
    struct TraceParams
    {
        public Vector4 originTan;   // xyz: ray origin (object space), w: tan(fovY/2)
        public Vector4 rightAspect; // xyz: camera right (object space), w: aspect
        public Vector4 up;          // xyz: camera up (object space)
        public Vector4 forward;     // xyz: camera forward (object space)
        public uint width, height, vertexStride, posOffset;
        public uint indexFormat;    // 0: UInt16, 1: UInt32
        public uint pad0, pad1, pad2;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct ProbeResult
    {
        public float hit, distance, primitiveIndex, pad;
        public Vector2 barycentric, pad2;
    }

    const string PluginName = "MetalRTTest";

    [DllImport(PluginName)] static extern IntPtr MetalRT_GetLastError();
    [DllImport(PluginName)] static extern int MetalRT_DeviceSupportsRaytracing();
    [DllImport(PluginName)] static extern int MetalRT_BuildAccelerationStructure
      (IntPtr vertexBuffer, uint vertexStride, uint positionOffset,
       IntPtr indexBuffer, uint indexFormat, uint indexByteOffset,
       uint triangleCount);
    [DllImport(PluginName)] static extern int MetalRT_TraceImage
      (in TraceParams traceParams, byte[] pixels);
    [DllImport(PluginName)] static extern int MetalRT_TraceProbes
      (in TraceParams traceParams, Vector4[] rays, int count,
       [Out] ProbeResult[] results);
    [DllImport(PluginName)] static extern void MetalRT_Dispose();

    static string LastError
      => Marshal.PtrToStringAnsi(MetalRT_GetLastError());

    // Scene bootstrap

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Bootstrap()
      => new GameObject("Metal RT Test", typeof(MetalRayTracingTest));

    // Probe rays in object space with analytic expectations for a torus of
    // major radius 1.0 / minor radius 0.4 lying in the XZ plane.
    static readonly (Vector3 origin, Vector3 dir, bool hit, float dist)[]
      Probes =
    {
        (Vector3.zero, Vector3.right, true, 0.6f),          // inner tube wall
        (new Vector3(1, 5, 0), Vector3.down, true, 4.6f),   // top of the tube
        (new Vector3(0, 5, 0), Vector3.down, false, 0),     // through the hole
        (new Vector3(5, 0, 0), Vector3.right, false, 0),    // pointing away
        (new Vector3(-3, 0, 0), Vector3.right, true, 1.6f), // outer wall
    };

    const float ProbeTolerance = 0.02f;

    // Private members

    Camera _camera;
    Transform _target;
    Mesh _mesh;
    Texture2D _result;
    TraceParams _params;

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

        SetUpScene();
        if (!BuildAccelerationStructure()) return;
        RunProbeTest();
        RunImageTest();
    }

    void OnDestroy() => MetalRT_Dispose();

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

    void SetUpScene()
    {
        _mesh = Resources.Load<Mesh>("Torus");
        Log($"Mesh: {_mesh.vertexCount} verts, isReadable = " +
            $"{_mesh.isReadable} (expected False)");

        var go = new GameObject("Torus", typeof(MeshFilter),
                                typeof(MeshRenderer));
        go.transform.rotation = Quaternion.Euler(25, 30, 0);
        go.GetComponent<MeshFilter>().sharedMesh = _mesh;
        go.GetComponent<MeshRenderer>().sharedMaterial =
          new Material(Shader.Find("MetalRT/ObjectNormal"));
        _target = go.transform;

        _camera = Camera.main;
        if (_camera == null)
            _camera = new GameObject("Camera", typeof(Camera))
                        .GetComponent<Camera>();
        _camera.transform.position = new Vector3(2.6f, 2.1f, -3.0f);
        _camera.transform.LookAt(Vector3.zero);
        _camera.fieldOfView = 40;
        _camera.rect = new Rect(0, 0, 0.5f, 1); // left half: raster reference
        _camera.clearFlags = CameraClearFlags.SolidColor;
        _camera.backgroundColor = new Color(0.1f, 0.1f, 0.2f);
    }

    // Acceleration structure construction from the non-readable mesh

    bool BuildAccelerationStructure()
    {
        var posStream = _mesh.GetVertexAttributeStream(VertexAttribute.Position);
        var posOffset = _mesh.GetVertexAttributeOffset(VertexAttribute.Position);
        var posFormat = _mesh.GetVertexAttributeFormat(VertexAttribute.Position);
        var posDim = _mesh.GetVertexAttributeDimension(VertexAttribute.Position);
        var stride = _mesh.GetVertexBufferStride(posStream);

        if (posFormat != VertexAttributeFormat.Float32 || posDim != 3)
        {
            Log($"FAIL: Unsupported position format {posFormat}x{posDim}");
            return false;
        }

        if (_mesh.GetTopology(0) != MeshTopology.Triangles ||
            _mesh.GetBaseVertex(0) != 0)
        {
            Log("FAIL: Unsupported mesh topology or base vertex");
            return false;
        }

        var indexCount = _mesh.GetIndexCount(0);
        var indexStart = _mesh.GetIndexStart(0);
        var is16Bit = _mesh.indexFormat == IndexFormat.UInt16;
        var indexSize = is16Bit ? 2u : 4u;

        Log($"Vertex layout: stream {posStream}, stride {stride}, " +
            $"position offset {posOffset}; {indexCount / 3} triangles, " +
            $"{_mesh.indexFormat} indices");

        // Native buffer pointers are only guaranteed valid within this frame,
        // so the acceleration structure is built right away.
        var vb = _mesh.GetNativeVertexBufferPtr(posStream);
        var ib = _mesh.GetNativeIndexBufferPtr();

        var ret = MetalRT_BuildAccelerationStructure
          (vb, (uint)stride, (uint)posOffset,
           ib, is16Bit ? 0u : 1u, indexStart * indexSize, indexCount / 3);

        if (ret != 0)
        {
            Log($"FAIL: Acceleration structure build error {ret}: {LastError}");
            return false;
        }

        Log("Acceleration structure built from non-readable mesh: OK");

        _params = new TraceParams
        {
            vertexStride = (uint)stride,
            posOffset = (uint)posOffset,
            indexFormat = is16Bit ? 0u : 1u
        };
        return true;
    }

    // Data level test: analytically verifiable probe rays

    void RunProbeTest()
    {
        var rays = new Vector4[Probes.Length * 2];
        for (var i = 0; i < Probes.Length; i++)
        {
            rays[i * 2] = Probes[i].origin;
            rays[i * 2 + 1] = Probes[i].dir;
        }

        var results = new ProbeResult[Probes.Length];
        var ret = MetalRT_TraceProbes(_params, rays, Probes.Length, results);
        if (ret != 0)
        {
            Log($"FAIL: Probe trace error {ret}: {LastError}");
            return;
        }

        var passed = 0;
        for (var i = 0; i < Probes.Length; i++)
        {
            var (origin, dir, expHit, expDist) = Probes[i];
            var r = results[i];
            var hit = r.hit > 0.5f;
            var ok = hit == expHit &&
                     (!expHit || Mathf.Abs(r.distance - expDist) < ProbeTolerance);
            if (ok) passed++;

            var expected = expHit ? $"hit at t={expDist}" : "miss";
            var actual = hit ?
              $"hit at t={r.distance:F4} (tri {r.primitiveIndex}, " +
              $"bary {r.barycentric.x:F2}/{r.barycentric.y:F2})" : "miss";
            Log($"Probe {i}: {origin} -> {dir}: {actual}; " +
                $"expected {expected} ... {(ok ? "PASS" : "FAIL")}");
        }

        Log($"Probe test: {passed}/{Probes.Length} passed" +
            (passed == Probes.Length ? " -- ALL PASS" : " -- FAILURE"));
    }

    // Visual test: full frame trace with the scene camera

    void RunImageTest()
    {
        var (w, h) = (_camera.pixelWidth, _camera.pixelHeight);

        var w2l = _target.worldToLocalMatrix;
        var ct = _camera.transform;
        var tanFov = Mathf.Tan(_camera.fieldOfView * Mathf.Deg2Rad / 2);

        var p = _params;
        p.originTan = V4(w2l.MultiplyPoint(ct.position), tanFov);
        p.rightAspect = V4(w2l.MultiplyVector(ct.right), (float)w / h);
        p.up = V4(w2l.MultiplyVector(ct.up), 0);
        p.forward = V4(w2l.MultiplyVector(ct.forward), 0);
        (p.width, p.height) = ((uint)w, (uint)h);

        var pixels = new byte[w * h * 4];
        var ret = MetalRT_TraceImage(p, pixels);
        if (ret != 0)
        {
            Log($"FAIL: Image trace error {ret}: {LastError}");
            return;
        }

        _result = new Texture2D(w, h, TextureFormat.RGBA32, false);
        _result.LoadRawTextureData(pixels);
        _result.Apply();

        var path = Path.GetFullPath
          (Path.Combine(Application.dataPath, "..", "Output", "rt-result.png"));
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllBytes(path, _result.EncodeToPNG());
        Log($"Ray traced image ({w}x{h}) written to {path}");
    }

    static Vector4 V4(Vector3 v, float w) => new Vector4(v.x, v.y, v.z, w);
}

} // namespace MetalRTTest
