using System;
using System.Collections;
using System.IO;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;
using static MetalRTTest.MetalRTPlugin;

namespace MetalRTTest {

// Test harness for the Metal hardware ray tracing path tracer (stage 1:
// URP Lit materials only). Builds a static URP scene at runtime, registers
// its meshes and materials with the native plugin, runs analytic
// verification passes (probe rays, direct lighting, furnace test), then
// path-traces progressively on the render thread. Shows the URP rasterized
// view (left) and the path traced result (right) side by side.
public sealed class PathTracerTest : MonoBehaviour
{
    // Scene bootstrap

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Bootstrap()
      => new GameObject("Metal RT Test", typeof(PathTracerTest));

    // World-space probe rays with analytic expectations for the static
    // scene layout below (instance order: 0 floor, 1 torus, 2 sphere,
    // 3 small torus, 4 emissive sphere, 5 furnace test sphere).
    static readonly (Vector3 origin, Vector3 dir,
                     bool hit, float dist, int instance)[] Probes =
    {
        // inner tube wall of the big torus (center y 0.4)
        (new Vector3(0, 0.4f, 0), Vector3.right, true, 0.6f, 1),
        // through the torus hole down to the floor
        (new Vector3(0, 5, 0), Vector3.down, true, 5.0f, 0),
        // top of the metal sphere (center y 0.6 + radius 0.6)
        (new Vector3(2.2f, 5, 0.5f), Vector3.down, true, 3.8f, 2),
        // pointing away from everything
        (new Vector3(0, 1, -6), Vector3.back, false, 0, 0),
        // outer wall of the big torus
        (new Vector3(-5, 0.4f, 0), Vector3.right, true, 3.6f, 1),
    };

    const float ProbeTolerance = 0.02f;

    // Furnace test sphere, far above the scene (URP culls it; only the
    // virtual path tracer camera looks at it).
    static readonly Vector3 FurnaceCenter = new Vector3(0, 50, 0);
    const float FurnaceScale = 2; // base radius 0.5 -> radius 1

    const int TestFrames = 32;      // accumulation frames per analytic test
    const int SaveFrame = 300;      // progressive frame to save the PNG at
    const uint ProductionBounces = 5;

    // Private members

    Camera _camera;
    Light _light;
    (int mesh, int material, Transform transform)[] _instances;
    Mesh[] _meshes = new Mesh[0];
    Material[] _materials;
    RenderTexture _result;
    IntPtr _resultPtr;
    CommandBuffer _traceCommands;
    IntPtr _eventFunc;
    IntPtr[] _eventData;
    int _eventSlot;
    uint _frameIndex;
    int _tracedFrames;
    bool _resetQueued = true;
    TraceParams? _cameraOverride;
    FrameSettings _settings;

    const int EventRingSize = 4;

    static void Log(string message) => Debug.Log("[MetalRT] " + message);

    // MonoBehaviour implementation

    void Start()
    {
        Log($"Graphics device: {SystemInfo.graphicsDeviceType} " +
            $"({SystemInfo.graphicsDeviceName}), " +
            $"pipeline: {GraphicsSettings.currentRenderPipeline?.GetType().Name}, " +
            $"color space: {QualitySettings.activeColorSpace}");
        Log($"Native Metal device supportsRaytracing = " +
            $"{MetalRT_DeviceSupportsRaytracing()} (expected 1)");

        if (!SetUpScene()) return;
        if (!SetUpMaterials()) return;
        if (!BuildInstanceAS()) return;
        RunProbeTest();
        SetUpRenderTarget();
        SetUpResultView();
        StartCoroutine(RunTestSequence());
    }

    void Update()
    {
        if (_result == null) return;
        TraceFrame();
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
        GUI.Label(new Rect(10, 10, 300, 20), "URP Raster (reference)");
        GUI.Label(new Rect(Screen.width / 2 + 10, 10, 380, 20),
                  $"Metal RT Path Tracer ({_tracedFrames} frames)");
    }

    // Shows the linear result texture through URP itself (a second camera
    // rendering a full screen quad on the right half), so the color
    // conversion at display time is exactly the same as the raster
    // reference on the left half.
    void SetUpResultView()
    {
        const int layer = 31;

        var quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
        quad.name = "Result View";
        quad.layer = layer;
        quad.transform.position = new Vector3(1000, 0, 1); // out of the scene
        quad.transform.localScale = new Vector3(2 * _camera.aspect, 2, 1);
        Destroy(quad.GetComponent<Collider>());

        var mat = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
        mat.SetTexture("_BaseMap", _result);
        quad.GetComponent<MeshRenderer>().sharedMaterial = mat;

        var view = new GameObject("Result Camera", typeof(Camera))
                     .GetComponent<Camera>();
        view.transform.position = new Vector3(1000, 0, 0);
        view.orthographic = true;
        view.orthographicSize = 1;
        view.cullingMask = 1 << layer;
        view.rect = new Rect(0.5f, 0, 0.5f, 1); // right half
        view.clearFlags = CameraClearFlags.SolidColor;
        view.backgroundColor = Color.black;

        _camera.cullingMask &= ~(1 << layer);
    }

    // Test scene construction: a static arrangement of URP/Lit materials
    // on a floor, lit by one directional light and a flat ambient.

    bool SetUpScene()
    {
        var plane = LoadAndRegisterMesh("Plane");
        var torus = LoadAndRegisterMesh("Torus");
        var sphere = LoadAndRegisterMesh("Sphere");
        if (plane < 0 || torus < 0 || sphere < 0) return false;

        var floorMat = MakeLitMaterial(new Color(0.5f, 0.5f, 0.5f), 0, 0.2f);
        var torusMat = MakeLitMaterial(new Color(0.8f, 0.15f, 0.1f), 0, 0.3f);
        torusMat.SetTexture("_BaseMap", MakeCheckerTexture());
        var metalMat = MakeLitMaterial(new Color(0.95f, 0.93f, 0.9f), 1, 0.9f);
        var whiteMat = MakeLitMaterial(new Color(0.9f, 0.9f, 0.9f), 0, 0.2f);
        var glowMat = MakeLitMaterial(Color.black, 0, 0.5f);
        glowMat.EnableKeyword("_EMISSION");
        glowMat.SetColor("_EmissionColor", new Color(4, 3, 1.5f));
        var furnaceMat = MakeLitMaterial(new Color(0.735357f, 0.735357f,
                                                   0.735357f), 0, 0); // rho=0.5

        _materials = new[]
          { floorMat, torusMat, metalMat, whiteMat, glowMat, furnaceMat };

        // Initial transforms must match the probe expectations above.
        _instances = new[]
        {
            (plane, 0, Spawn("Floor", plane, floorMat,
              Vector3.zero, 1)),
            (torus, 1, Spawn("Torus", torus, torusMat,
              new Vector3(0, 0.4f, 0), 1)),
            (sphere, 2, Spawn("Metal Sphere", sphere, metalMat,
              new Vector3(2.2f, 0.6f, 0.5f), 1.2f)),
            (torus, 3, Spawn("Small Torus", torus, whiteMat,
              new Vector3(-2.1f, 0.2f, 0.8f), 0.5f)),
            (sphere, 4, Spawn("Glow Sphere", sphere, glowMat,
              new Vector3(-0.9f, 0.25f, -1.3f), 0.5f)),
            (sphere, 5, Spawn("Furnace Sphere", sphere, furnaceMat,
              FurnaceCenter, FurnaceScale)),
        };

        _light = new GameObject("Directional Light", typeof(Light))
                   .GetComponent<Light>();
        _light.type = LightType.Directional;
        _light.transform.rotation = Quaternion.Euler(50, -30, 0);
        _light.color = Color.white;
        _light.intensity = 1.5f;

        RenderSettings.ambientMode = AmbientMode.Flat;
        RenderSettings.ambientLight = new Color(0.15f, 0.17f, 0.22f);

        _camera = Camera.main;
        if (_camera == null)
            _camera = new GameObject("Camera", typeof(Camera))
                        .GetComponent<Camera>();
        _camera.transform.position = new Vector3(3.0f, 2.8f, -5.6f);
        _camera.transform.LookAt(new Vector3(0, 0.3f, 0));
        _camera.fieldOfView = 45;
        _camera.rect = new Rect(0, 0, 0.5f, 1); // left half: raster reference
        _camera.clearFlags = CameraClearFlags.SolidColor;
        _camera.backgroundColor = RenderSettings.ambientLight;

        _settings = ProductionSettings();
        return true;
    }

    FrameSettings ProductionSettings() => new FrameSettings
    {
        envColor = RenderSettings.ambientLight.linear,
        lightDir = _light.transform.forward,
        // Unity's punctual light convention folds 1/pi into the Lambert
        // term (diffuse = albedo * light * cos), so premultiply by pi for
        // the physically normalized BRDF used by the path tracer.
        lightColor = _light.color.linear * (_light.intensity * Mathf.PI),
        maxBounces = ProductionBounces,
        exposure = 1
    };

    static Material MakeLitMaterial(Color color, float metallic, float smoothness)
    {
        var mat = new Material(Shader.Find("Universal Render Pipeline/Lit"));
        mat.SetColor("_BaseColor", color);
        mat.SetFloat("_Metallic", metallic);
        mat.SetFloat("_Smoothness", smoothness);
        return mat;
    }

    static Texture2D MakeCheckerTexture()
    {
        const int size = 256, cells = 8;
        var tex = new Texture2D(size, size, TextureFormat.RGBA32, true);
        var pixels = new Color32[size * size];
        for (var y = 0; y < size; y++)
            for (var x = 0; x < size; x++)
            {
                var odd = (x * cells / size + y * cells / size) % 2 == 1;
                var v = (byte)(odd ? 230 : 100);
                pixels[y * size + x] = new Color32(v, v, v, 255);
            }
        tex.SetPixels32(pixels);
        tex.Apply(true, true); // upload mips, no longer readable
        return tex;
    }

    Transform Spawn(string name, int meshIndex, Material material,
                    Vector3 position, float scale)
    {
        var go = new GameObject(name, typeof(MeshFilter), typeof(MeshRenderer));
        go.transform.position = position;
        go.transform.localScale = Vector3.one * scale;
        go.GetComponent<MeshFilter>().sharedMesh = _meshes[meshIndex];
        go.GetComponent<MeshRenderer>().sharedMaterial = material;
        return go.transform;
    }

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

        // Normal and UV must live in the same vertex stream as position for
        // the single-buffer fetch in the shader.
        var normalOffset = AttributeOffset(mesh, VertexAttribute.Normal,
                                           posStream, 3);
        var uvOffset = AttributeOffset(mesh, VertexAttribute.TexCoord0,
                                       posStream, 2);

        var indexCount = mesh.GetIndexCount(0);
        var indexStart = mesh.GetIndexStart(0);
        var is16Bit = mesh.indexFormat == IndexFormat.UInt16;
        var indexSize = is16Bit ? 2u : 4u;

        // Native buffer pointers are only guaranteed valid within this
        // frame, so the BLAS is built right away (synchronously).
        var ret = MetalRT_AddMesh
          (mesh.GetNativeVertexBufferPtr(posStream), (uint)stride,
           (uint)posOffset, normalOffset, uvOffset,
           mesh.GetNativeIndexBufferPtr(),
           is16Bit ? 0u : 1u, indexStart * indexSize, indexCount / 3);

        if (ret < 0)
        {
            Log($"FAIL: BLAS build error {ret} for {resourceName}: {LastError}");
            return -1;
        }

        Log($"BLAS #{ret} built from non-readable mesh {resourceName} " +
            $"({indexCount / 3} triangles, normal@{normalOffset}, " +
            $"uv@{uvOffset}): OK");

        Array.Resize(ref _meshes, ret + 1);
        _meshes[ret] = mesh;
        return ret;
    }

    static uint AttributeOffset(Mesh mesh, VertexAttribute attr,
                                int requiredStream, int requiredDim)
    {
        if (!mesh.HasVertexAttribute(attr)) return NoAttribute;
        if (mesh.GetVertexAttributeStream(attr) != requiredStream ||
            mesh.GetVertexAttributeFormat(attr) != VertexAttributeFormat.Float32 ||
            mesh.GetVertexAttributeDimension(attr) != requiredDim)
            return NoAttribute;
        return (uint)mesh.GetVertexAttributeOffset(attr);
    }

    // Material table from URP/Lit material properties

    bool SetUpMaterials()
    {
        var descs = new MaterialDesc[_materials.Length];
        for (var i = 0; i < _materials.Length; i++)
        {
            var mat = _materials[i];
            var tex = mat.GetTexture("_BaseMap") as Texture2D;
            var emission = mat.IsKeywordEnabled("_EMISSION") ?
              mat.GetColor("_EmissionColor") : Color.black;
            descs[i] = new MaterialDesc
            {
                baseColor = mat.GetColor("_BaseColor").linear,
                emission = emission, // HDR colors are already linear
                baseMapST = mat.GetVector("_BaseMap_ST"),
                metallic = mat.GetFloat("_Metallic"),
                smoothness = mat.GetFloat("_Smoothness"),
                hasBaseMap = tex != null ? 1u : 0u,
                texture = tex != null ?
                  tex.GetNativeTexturePtr().ToInt64() : 0
            };
        }

        var ret = MetalRT_SetMaterials(descs, descs.Length);
        if (ret != 0)
        {
            Log($"FAIL: Material table error {ret}: {LastError}");
            return false;
        }

        Log($"Material table built with {descs.Length} URP/Lit materials: OK");
        return true;
    }

    // TLAS instance descriptors from the scene transforms

    InstanceDesc[] MakeInstanceDescs()
    {
        var descs = new InstanceDesc[_instances.Length];
        for (var i = 0; i < _instances.Length; i++)
        {
            var (meshIndex, materialIndex, transform) = _instances[i];
            var l2w = transform.localToWorldMatrix;
            var nrm = l2w.inverse.transpose;
            descs[i] = new InstanceDesc
            {
                meshIndex = meshIndex,
                materialIndex = materialIndex,
                objectToWorld0 = l2w.GetRow(0),
                objectToWorld1 = l2w.GetRow(1),
                objectToWorld2 = l2w.GetRow(2),
                normalMatrix0 = nrm.GetRow(0),
                normalMatrix1 = nrm.GetRow(1),
                normalMatrix2 = nrm.GetRow(2)
            };
        }
        return descs;
    }

    // Synchronous TLAS build used by the probe test path.
    bool BuildInstanceAS()
    {
        var descs = MakeInstanceDescs();
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

    // Data level test 0: analytically verifiable probe rays

    void RunProbeTest()
    {
        var rays = new Vector4[Probes.Length * 2];
        for (var i = 0; i < Probes.Length; i++)
        {
            rays[i * 2] = Probes[i].origin;
            rays[i * 2 + 1] = Probes[i].dir;
        }

        var results = new ProbeResult[Probes.Length];
        var ret = MetalRT_TraceProbes(rays, Probes.Length, results);
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
              $"hit at t={r.distance:F4} on instance {(int)r.instanceIndex}" :
              "miss";
            Log($"Probe {i}: {actual}; expected {expected} ... " +
                (ok ? "PASS" : "FAIL"));
        }

        Log($"Probe test: {passed}/{Probes.Length} passed" +
            (passed == Probes.Length ? " -- ALL PASS" : " -- FAILURE"));
    }

    // Per-frame trace on the render thread via IssuePluginEventAndData

    void SetUpRenderTarget()
    {
        var (w, h) = (_camera.pixelWidth, _camera.pixelHeight);
        _result = new RenderTexture(w, h, 0, RenderTextureFormat.ARGBFloat)
          { enableRandomWrite = true };
        _result.Create();
        _resultPtr = _result.GetNativeTexturePtr();
        Log($"RenderTexture ({w}x{h}, ARGBFloat) created; native ptr acquired");

        _eventFunc = MetalRT_GetRenderEventFunc();
        _traceCommands = new CommandBuffer { name = "MetalRT Trace" };

        // A small ring of unmanaged blobs so the render thread never reads
        // a blob the main thread is currently rewriting.
        _eventData = new IntPtr[EventRingSize];
        for (var i = 0; i < EventRingSize; i++)
            _eventData[i] = Marshal.AllocHGlobal(EventDataSize);
    }

    TraceParams CameraParams()
    {
        var ct = _camera.transform;
        var tanFov = Mathf.Tan(_camera.fieldOfView * Mathf.Deg2Rad / 2);
        var (w, h) = (_result.width, _result.height);
        return new TraceParams
        {
            originTan = V4(ct.position, tanFov),
            rightAspect = V4(ct.right, (float)w / h),
            up = V4(ct.up, 0),
            forward = V4(ct.forward, 0),
            width = (uint)w,
            height = (uint)h
        };
    }

    void TraceFrame()
    {
        var p = _cameraOverride ?? CameraParams();
        var s = _settings;
        s.reset = _resetQueued;
        _resetQueued = false;

        _eventSlot = (_eventSlot + 1) % EventRingSize;
        WriteEventData(_eventData[_eventSlot], p, _resultPtr, _frameIndex++,
                       s, MakeInstanceDescs());

        _traceCommands.Clear();
        _traceCommands.IssuePluginEventAndData
          (_eventFunc, 0, _eventData[_eventSlot]);
        Graphics.ExecuteCommandBuffer(_traceCommands);
        _tracedFrames++;
    }

    // Analytic radiance tests (T1 direct lighting, T2 furnace), then the
    // production progressive render.

    IEnumerator RunTestSequence()
    {
        // T1: direct lighting on the floor. env off, single bounce,
        // diffuse-only BSDF, linear output.
        var t1Point = new Vector3(1.5f, 0, -1.8f);
        _settings = ProductionSettings();
        _settings.envColor = Color.black;
        _settings.maxBounces = 1;
        _settings.linearOutput = true;
        _settings.debugFlags = 1; // diffuse only
        _resetQueued = true;

        for (var i = 0; i < TestFrames; i++) yield return null;

        {
            var albedo = _materials[0].GetColor("_BaseColor").linear;
            var cos = Vector3.Dot(Vector3.up, -_settings.lightDir);
            var expected = (Color)(albedo * _settings.lightColor) *
                           (cos / Mathf.PI);
            var measured = ReadResultAverage(WorldToResultPixel(t1Point), 2);
            CheckClose("T1 direct lighting (floor)", measured, expected, 0.03f);
        }

        // T2: furnace test on a convex Lambertian sphere in a uniform
        // environment. Every path returns exactly rho * env.
        var env = new Color(0.5f, 0.5f, 0.5f);
        _settings = ProductionSettings();
        _settings.envColor = env;
        _settings.lightColor = Color.black;
        _settings.maxBounces = 2;
        _settings.linearOutput = true;
        _settings.debugFlags = 1; // diffuse only
        _cameraOverride = LookAtParams(FurnaceCenter + new Vector3(0, 0, -4),
                                       FurnaceCenter);
        _resetQueued = true;

        for (var i = 0; i < TestFrames; i++) yield return null;

        {
            var rho = _materials[5].GetColor("_BaseColor").linear;
            var expected = (Color)(rho * env);
            var center = new Vector2Int(_result.width / 2, _result.height / 2);
            var measured = ReadResultAverage(center, 2);
            CheckClose("T2 furnace (rho=0.5 sphere)", measured, expected, 0.02f);
        }

        // Production: progressive path tracing of the visible scene.
        _settings = ProductionSettings();
        _cameraOverride = null;
        _resetQueued = true;
        _tracedFrames = 0;

        while (_tracedFrames < SaveFrame) yield return null;

        Log($"Render thread events executed: {MetalRT_GetEventFrameCount()} " +
            $"(total issued: {_frameIndex})");
        SaveResultImage();
    }

    Vector2Int WorldToResultPixel(Vector3 worldPos)
    {
        var sp = _camera.WorldToScreenPoint(worldPos);
        var p = new Vector2Int(Mathf.RoundToInt(sp.x), Mathf.RoundToInt(sp.y));
        if (sp.z <= 0 || p.x < 2 || p.y < 2 ||
            p.x >= _result.width - 2 || p.y >= _result.height - 2)
            Log($"WARNING: test point {worldPos} projects off screen ({p})");
        return p;
    }

    TraceParams LookAtParams(Vector3 position, Vector3 target)
    {
        var fwd = (target - position).normalized;
        var right = Vector3.Cross(Vector3.up, fwd).normalized;
        var up = Vector3.Cross(fwd, right);
        var (w, h) = (_result.width, _result.height);
        return new TraceParams
        {
            originTan = V4(position, Mathf.Tan(45 * Mathf.Deg2Rad / 2)),
            rightAspect = V4(right, (float)w / h),
            up = V4(up, 0),
            forward = V4(fwd, 0),
            width = (uint)w,
            height = (uint)h
        };
    }

    // Reads back a small region of the (linear output mode) result texture
    // and returns the average color. ReadPixels flushes the render thread,
    // so the last issued frame is included.
    Color ReadResultAverage(Vector2Int center, int radius)
    {
        var n = radius * 2 + 1;
        var prev = RenderTexture.active;
        RenderTexture.active = _result;
        var tex = new Texture2D(n, n, TextureFormat.RGBAFloat, false);
        tex.ReadPixels(new Rect(center.x - radius, center.y - radius, n, n),
                       0, 0);
        tex.Apply();
        RenderTexture.active = prev;

        var sum = Color.black;
        foreach (var c in tex.GetPixels()) sum += c;
        Destroy(tex);
        return sum / (n * n);
    }

    static void CheckClose(string name, Color measured, Color expected,
                           float tolerance)
    {
        var maxRef = Mathf.Max(expected.maxColorComponent, 1e-4f);
        var err = Mathf.Max(Mathf.Abs(measured.r - expected.r),
                            Mathf.Abs(measured.g - expected.g),
                            Mathf.Abs(measured.b - expected.b)) / maxRef;
        Log($"{name}: measured ({measured.r:F4}, {measured.g:F4}, " +
            $"{measured.b:F4}), expected ({expected.r:F4}, {expected.g:F4}, " +
            $"{expected.b:F4}), rel. error {err:P2} ... " +
            (err < tolerance ? "PASS" : "FAIL"));
    }

    void SaveResultImage()
    {
        var (w, h) = (_result.width, _result.height);
        var prev = RenderTexture.active;
        RenderTexture.active = _result;
        var tex = new Texture2D(w, h, TextureFormat.RGBAFloat, false);
        tex.ReadPixels(new Rect(0, 0, w, h), 0, 0);
        RenderTexture.active = prev;

        // The result texture holds linear values; encode to sRGB for PNG.
        var pixels = tex.GetPixels();
        for (var i = 0; i < pixels.Length; i++)
        {
            var g = pixels[i].gamma;
            g.a = 1;
            pixels[i] = g;
        }
        tex.SetPixels(pixels);
        tex.Apply();

        var path = Path.GetFullPath
          (Path.Combine(Application.dataPath, "..", "Output", "rt-result.png"));
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllBytes(path, tex.EncodeToPNG());
        Destroy(tex);
        Log($"Path traced image ({w}x{h}, {_tracedFrames} frames) " +
            $"written to {path}");
    }

    static Vector4 V4(Vector3 v, float w) => new Vector4(v.x, v.y, v.z, w);
}

} // namespace MetalRTTest
