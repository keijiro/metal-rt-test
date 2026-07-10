using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using static MetalRTTest.MetalRTPlugin;

namespace MetalRTTest {

// Test harness for the Metal RT path tracer. Builds a static URP scene at
// runtime and lets MetalRTSceneRegistry auto-register everything from the
// scene's MeshRenderers. Rendering happens through URP: the left camera
// rasterizes the scene normally, the right camera uses the renderer with
// MetalRTPathTracerFeature. Runs analytic verification passes (probe rays,
// T1-T7) before the production progressive render.
public sealed class PathTracerTest : MonoBehaviour
{
    // Scene bootstrap

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Bootstrap()
      => new GameObject("Metal RT Test", typeof(PathTracerTest));

    const float ProbeTolerance = 0.02f;

    // Off-scene test objects (only the analytic tests' virtual camera
    // looks at them).
    static readonly Vector3 FurnaceCenter = new Vector3(0, 50, 0);
    const float FurnaceScale = 2; // base radius 0.5 -> radius 1
    static readonly Vector3 VertexColorCenter = new Vector3(30, 50, 0);
    static readonly Vector3 CutoutCenter = new Vector3(60, 50, 0);
    static readonly Vector3 PunctualCenter = new Vector3(90, 50, 0);
    static readonly Vector3 ShadowCenter = new Vector3(120, 50, 0);
    const float TestQuadScale = 2; // quad extent +-1 -> +-2

    static readonly Color32[] QuadCorners =
    {
        new Color32(255, 0, 0, 255), new Color32(0, 255, 0, 255),
        new Color32(0, 0, 255, 255), new Color32(255, 255, 255, 255)
    };

    const int TestFrames = 32;   // accumulation frames per analytic test
    const int SaveFrame = 300;   // progressive frame to save the PNG at
    const uint ProductionBounces = 5;

    // Procedural test material parameters (linear colors)
    static readonly Color ProcColorA = new Color(0.9f, 0.25f, 0.1f);
    static readonly Color ProcColorB = new Color(0.1f, 0.3f, 0.9f);
    const float ProcCheckerScale = 6;

    // Private members

    Camera _camera;   // left half: URP raster reference
    Camera _ptCamera; // right half: URP camera with the path tracer feature
    Light _light;
    Light _pointLight, _spotLight; // T8 test lights (disabled by default)
    Light _shadowLight;            // T9 test light (disabled by default)
    MetalRTSceneRegistry _registry;

    Material _floorMat, _torusMat, _metalMat, _whiteMat, _glowMat,
             _furnaceMat, _vcMat, _cutoutMat, _punctualMat,
             _shadowRecvMat, _shadowOccMat;
    Transform _floorT, _torusT, _metalT, _whiteT, _glowT, _furnaceT,
              _vcT, _cutoutT, _punctualT;

    MetalRTPathTracer.MaterialCompute _procedural;
    MetalRTPathTracer.MaterialCompute _sgCompute;
    int _tracedFrames;

    static MetalRTPathTracer Tracer => MetalRTPathTracer.Instance;

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

        SetUpScene();

        _registry = new MetalRTSceneRegistry();
        if (!_registry.Build(Tracer)) return;
        if (!BuildInstanceAS()) return;
        RunProbeTest();
        if (!SetUpTracer()) return;
        StartCoroutine(RunTestSequence());
    }

    void Update()
    {
        if (Tracer.IsConfigured) _tracedFrames++;
    }

    void OnDestroy()
    {
        Tracer.Dispose();
        MetalRT_Dispose();
    }

    void OnGUI()
    {
        if (!Tracer.IsConfigured) return;
        GUI.Label(new Rect(10, 10, 300, 20), "URP Raster (reference)");
        GUI.Label(new Rect(Screen.width / 2 + 10, 10, 380, 20),
                  "Metal RT Path Tracer via RendererFeature " +
                  $"({_tracedFrames} frames)");
    }

    // Test scene construction (registration happens automatically through
    // MetalRTSceneRegistry afterwards)

    void SetUpScene()
    {
        // Remove any pre-existing lights (e.g., the default scene light) so
        // the light set is fully under the harness's control — the path
        // tracer now consumes every enabled punctual light in the scene.
        foreach (var l in FindObjectsByType<Light>(FindObjectsSortMode.None))
            DestroyImmediate(l.gameObject);

        var plane = Resources.Load<Mesh>("Plane");
        var torus = Resources.Load<Mesh>("Torus");
        var sphere = Resources.Load<Mesh>("Sphere");
        var quad = MakeQuadMesh(QuadCorners);

        // The floor uses the test Shader Graph: URP rasterizes it with the
        // graph's own shader while the path tracer evaluates the compute
        // shader generated from the same graph, so both views must match.
        _floorMat = MakeShaderGraphMaterial();
        _torusMat = MakeLitMaterial(new Color(0.8f, 0.15f, 0.1f), 0, 0.3f);
        _torusMat.SetTexture("_BaseMap", MakeCheckerTexture());
        _metalMat = MakeLitMaterial(new Color(0.95f, 0.93f, 0.9f), 1, 0.9f);
        _whiteMat = MakeLitMaterial(new Color(0.9f, 0.9f, 0.9f), 0, 0.2f);
        _glowMat = MakeLitMaterial(Color.black, 0, 0.5f);
        _glowMat.EnableKeyword("_EMISSION");
        _glowMat.SetColor("_EmissionColor", new Color(4, 3, 1.5f));
        _furnaceMat = MakeLitMaterial(new Color(0.735357f, 0.735357f,
                                                0.735357f), 0, 0); // rho=0.5
        // Magenta fallbacks: visible if the material computes fail to run.
        _vcMat = MakeLitMaterial(Color.magenta, 0, 0.2f);
        _cutoutMat = MakeLitMaterial(Color.magenta, 0, 0.2f);

        _floorT = Spawn("Floor", plane, _floorMat, Vector3.zero, 1);
        _torusT = Spawn("Torus", torus, _torusMat, new Vector3(0, 0.4f, 0), 1);
        _metalT = Spawn("Metal Sphere", sphere, _metalMat,
                        new Vector3(2.2f, 0.6f, 0.5f), 1.2f);
        _whiteT = Spawn("Small Torus", torus, _whiteMat,
                        new Vector3(-2.1f, 0.2f, 0.8f), 0.5f);
        _glowT = Spawn("Glow Sphere", sphere, _glowMat,
                       new Vector3(-0.9f, 0.25f, -1.3f), 0.5f);
        _furnaceT = Spawn("Furnace Sphere", sphere, _furnaceMat,
                          FurnaceCenter, FurnaceScale);
        _vcT = Spawn("Vertex Color Quad", quad, _vcMat,
                     VertexColorCenter, TestQuadScale);
        _cutoutT = Spawn("Cutout Quad", quad, _cutoutMat,
                         CutoutCenter, TestQuadScale);

        // Punctual light receiver (T8): solid gray quad facing -Z with a
        // point and a spot light in front (disabled until the test).
        _punctualMat = MakeLitMaterial(new Color(0.5f, 0.5f, 0.5f), 0, 0.2f);
        _punctualT = Spawn("Punctual Quad", quad, _punctualMat,
                           PunctualCenter, TestQuadScale);

        // Alpha-tested shadow setup (T9): a gray receiver with a small
        // alpha-clipped occluder next to a point light. The occluder's
        // base map alpha is 0.5; the test swaps _Cutoff around it.
        _shadowRecvMat = MakeLitMaterial(new Color(0.5f, 0.5f, 0.5f), 0, 0.2f);
        Spawn("Shadow Receiver", quad, _shadowRecvMat,
              ShadowCenter, TestQuadScale);
        _shadowOccMat = MakeLitMaterial(Color.white, 0, 0.2f);
        _shadowOccMat.EnableKeyword("_ALPHATEST_ON");
        _shadowOccMat.SetFloat("_Cutoff", 0.8f);
        var halfAlpha = new Texture2D(1, 1, TextureFormat.RGBA32, false);
        halfAlpha.SetPixel(0, 0, new Color(1, 1, 1, 0.5f));
        halfAlpha.Apply(false, true);
        _shadowOccMat.SetTexture("_BaseMap", halfAlpha);
        Spawn("Shadow Occluder", quad, _shadowOccMat,
              ShadowCenter + new Vector3(0, 0, -2.7f), 0.3f);

        _light = new GameObject("Directional Light", typeof(Light))
                   .GetComponent<Light>();
        _light.type = LightType.Directional;
        _light.transform.rotation = Quaternion.Euler(50, -30, 0);
        _light.color = Color.white;
        _light.intensity = 1.5f;

        _pointLight = new GameObject("Test Point Light", typeof(Light))
                        .GetComponent<Light>();
        _pointLight.type = LightType.Point;
        _pointLight.transform.position = PunctualCenter + new Vector3(0, 0, -3);
        _pointLight.range = 10;
        _pointLight.intensity = 2;
        _pointLight.enabled = false;

        _spotLight = new GameObject("Test Spot Light", typeof(Light))
                       .GetComponent<Light>();
        _spotLight.type = LightType.Spot;
        _spotLight.transform.position = PunctualCenter + new Vector3(0, 0, -3);
        _spotLight.transform.rotation = Quaternion.identity; // aims +Z
        _spotLight.range = 10;
        _spotLight.intensity = 2;
        _spotLight.spotAngle = 60;
        _spotLight.innerSpotAngle = 40;
        _spotLight.enabled = false;

        _shadowLight = new GameObject("Test Shadow Light", typeof(Light))
                         .GetComponent<Light>();
        _shadowLight.type = LightType.Point;
        _shadowLight.transform.position = ShadowCenter + new Vector3(0, 0, -3);
        _shadowLight.range = 10;
        _shadowLight.intensity = 2;
        _shadowLight.enabled = false;

        RenderSettings.ambientMode = AmbientMode.Flat;
        RenderSettings.ambientLight = new Color(0.15f, 0.17f, 0.22f);

        _camera = Camera.main;
        if (_camera == null)
            _camera = new GameObject("Camera", typeof(Camera))
                        .GetComponent<Camera>();
        _camera.transform.position = new Vector3(3.0f, 2.8f, -5.6f);
        _camera.transform.LookAt(new Vector3(0, 0.3f, 0));
        _camera.fieldOfView = 45;
        _camera.rect = new Rect(0, 0, 0.5f, 1); // left half
        _camera.clearFlags = CameraClearFlags.SolidColor;
        _camera.backgroundColor = RenderSettings.ambientLight;

        // Right half: same view, rendered by the path tracer feature
        // (renderer index 1). The raster pass is skipped via culling mask.
        _ptCamera = new GameObject("PT Camera", typeof(Camera))
                      .GetComponent<Camera>();
        _ptCamera.transform.SetPositionAndRotation
          (_camera.transform.position, _camera.transform.rotation);
        _ptCamera.fieldOfView = _camera.fieldOfView;
        _ptCamera.rect = new Rect(0.5f, 0, 0.5f, 1);
        _ptCamera.cullingMask = 0;
        _ptCamera.clearFlags = CameraClearFlags.SolidColor;
        _ptCamera.backgroundColor = Color.black;
        _ptCamera.GetUniversalAdditionalCameraData().SetRenderer(1);
    }

    FrameSettings ProductionSettings() => new FrameSettings
    {
        envColor = RenderSettings.ambientLight.linear,
        lights = CollectLights(),
        maxBounces = ProductionBounces,
        exposure = 1
    };

    // Gathers the scene's enabled punctual lights (up to MaxLights).
    static LightDesc[] CollectLights()
    {
        var lights = FindObjectsByType<Light>(FindObjectsSortMode.InstanceID);
        var list = new System.Collections.Generic.List<LightDesc>();
        foreach (var l in lights)
        {
            if (!l.isActiveAndEnabled || list.Count >= MaxLights) continue;
            if (l.type != LightType.Directional && l.type != LightType.Point &&
                l.type != LightType.Spot) continue;
            list.Add(MakeLightDesc(l));
        }
        return list.ToArray();
    }

    // Directional light contribution used by floor test expectations.
    (Vector3 toLight, Color color) DirLight()
      => (-_light.transform.forward,
          _light.color.linear * (_light.intensity * Mathf.PI));

    Material MakeShaderGraphMaterial()
    {
        var mat = new Material(Shader.Find("Lit/TestGraph"));
        mat.SetColor("_BaseColor", Color.white);
        mat.SetFloat("_Smoothness", 0.3f);
        mat.SetTexture("_BaseMap", MakeCheckerTexture());

        // Non-metallic, full-gloss-scale MOS map (R: metallic 0, A: 1)
        var mos = new Texture2D(1, 1, TextureFormat.RGBA32, false);
        mos.SetPixel(0, 0, new Color(0, 0, 0, 1));
        mos.Apply(false, true);
        mat.SetTexture("_MetallicGlossMap", mos);
        return mat;
    }

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

    // Runtime-built quad (extent +-1 in XY, facing -Z) with vertex colors
    // and UVs, made non-readable on upload.
    static Mesh MakeQuadMesh(Color32[] cornerColors)
    {
        var mesh = new Mesh { name = "RuntimeQuad" };
        mesh.vertices = new[]
        {
            new Vector3(-1, -1, 0), new Vector3(1, -1, 0),
            new Vector3(1, 1, 0), new Vector3(-1, 1, 0)
        };
        mesh.normals = new[]
        {
            new Vector3(0, 0, -1), new Vector3(0, 0, -1),
            new Vector3(0, 0, -1), new Vector3(0, 0, -1)
        };
        mesh.uv = new[]
        {
            new Vector2(0, 0), new Vector2(1, 0),
            new Vector2(1, 1), new Vector2(0, 1)
        };
        mesh.colors32 = cornerColors;
        mesh.triangles = new[] { 0, 1, 2, 0, 2, 3 };
        mesh.UploadMeshData(true); // no longer readable
        return mesh;
    }

    Transform Spawn(string name, Mesh mesh, Material material,
                    Vector3 position, float scale)
    {
        var go = new GameObject(name, typeof(MeshFilter), typeof(MeshRenderer));
        go.transform.position = position;
        go.transform.localScale = Vector3.one * scale;
        go.GetComponent<MeshFilter>().sharedMesh = mesh;
        go.GetComponent<MeshRenderer>().sharedMaterial = material;
        return go.transform;
    }

    // Synchronous TLAS build used by the probe test path.
    bool BuildInstanceAS()
    {
        var descs = _registry.MakeDescs();
        var ret = MetalRT_BuildInstanceAS(descs, descs.Length);
        if (ret != 0)
        {
            Log($"FAIL: Instance AS build error {ret}: {LastError}");
            return false;
        }
        Log($"Instance AS (TLAS) built with {descs.Length} instances: OK");
        return true;
    }

    // Data level test 0: analytically verifiable probe rays

    void RunProbeTest()
    {
        // World-space probes with analytic expectations; hit instances are
        // resolved through the registry (auto-registration order).
        var probes = new (Vector3 origin, Vector3 dir,
                          bool hit, float dist, Transform inst)[]
        {
            (new Vector3(0, 0.4f, 0), Vector3.right, true, 0.6f, _torusT),
            (new Vector3(0, 5, 0), Vector3.down, true, 5.0f, _floorT),
            (new Vector3(2.2f, 5, 0.5f), Vector3.down, true, 3.8f, _metalT),
            (new Vector3(0, 1, -6), Vector3.back, false, 0, null),
            (new Vector3(-5, 0.4f, 0), Vector3.right, true, 3.6f, _torusT),
        };

        var rays = new Vector4[probes.Length * 2];
        for (var i = 0; i < probes.Length; i++)
        {
            rays[i * 2] = probes[i].origin;
            rays[i * 2 + 1] = probes[i].dir;
        }

        var results = new ProbeResult[probes.Length];
        var ret = MetalRT_TraceProbes(rays, probes.Length, results);
        if (ret != 0)
        {
            Log($"FAIL: Probe trace error {ret}: {LastError}");
            return;
        }

        var passed = 0;
        for (var i = 0; i < probes.Length; i++)
        {
            var (origin, dir, expHit, expDist, inst) = probes[i];
            var expInst = inst != null ? _registry.InstanceIndexOf(inst) : -1;
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

        Log($"Probe test: {passed}/{probes.Length} passed" +
            (passed == probes.Length ? " -- ALL PASS" : " -- FAILURE"));
    }

    // Tracer configuration (buffers, hand-written material computes; the
    // Shader Graph compute for the floor was attached by the registry)

    bool SetUpTracer()
    {
        Tracer.Configure(_registry.MakeDescs);
        Tracer.Settings = ProductionSettings();
        if (!Tracer.EnsureResources(_ptCamera.pixelWidth,
                                    _ptCamera.pixelHeight))
        {
            Log("FAIL: tracer resource setup");
            return false;
        }
        Log($"Tracer configured ({_ptCamera.pixelWidth}x" +
            $"{_ptCamera.pixelHeight}); driven by the URP renderer feature");

        _sgCompute = _registry.ComputeOf(_floorMat);
        if (_sgCompute == null)
        {
            Log("FAIL: Shader Graph compute missing for the floor material");
            return false;
        }

        var procCs = Resources.Load<ComputeShader>("TestProcedural");
        _procedural = new MetalRTPathTracer.MaterialCompute
        {
            Shader = procCs,
            Kernel = procCs.FindKernel("EvaluateMaterial"),
            MaterialIndex = _registry.MaterialIndexOf(_whiteMat),
            Bind = cb =>
            {
                cb.SetComputeVectorParam(procCs, "_ColorA", ProcColorA);
                cb.SetComputeVectorParam(procCs, "_ColorB", ProcColorB);
                cb.SetComputeFloatParam(procCs, "_CheckerScale",
                                        ProcCheckerScale);
            }
        };
        Tracer.MaterialComputes.Add(_procedural);

        var vcCs = Resources.Load<ComputeShader>("TestVertexColor");
        var vcMat = _vcMat;
        Tracer.MaterialComputes.Add(new MetalRTPathTracer.MaterialCompute
          { Shader = vcCs, Kernel = vcCs.FindKernel("EvaluateMaterial"),
            MaterialIndex = _registry.MaterialIndexOf(_vcMat),
            Bind = cb => MetalRTSceneRegistry.BindMaterialKeywords
                           (cb, vcCs, vcMat) });

        var cutCs = Resources.Load<ComputeShader>("TestCutout");
        Tracer.MaterialComputes.Add(new MetalRTPathTracer.MaterialCompute
          { Shader = cutCs, Kernel = cutCs.FindKernel("EvaluateMaterial"),
            MaterialIndex = _registry.MaterialIndexOf(_cutoutMat) });

        Log("Hand-written material computes registered (procedural, " +
            "vertex color, cutout)");
        return true;
    }

    // Analytic radiance tests, then the production progressive render.

    IEnumerator RunTestSequence()
    {
        var sgCompute = _sgCompute;

        // Expected linear floor albedo at a world point: checker base map
        // sample times the base color tint (cell interiors only).
        Color FloorAlbedo(Vector3 p)
        {
            var uv = new Vector2((5 - p.x) / 10, (p.z + 5) / 10);
            var cell = uv * 8; // checker texture cells
            var odd = (Mathf.FloorToInt(cell.x) + Mathf.FloorToInt(cell.y))
                      % 2 == 1;
            var texel = Mathf.GammaToLinearSpace((odd ? 230 : 100) / 255f);
            return texel * _floorMat.GetColor("_BaseColor").linear;
        }

        // T1: direct lighting on the floor through the native URP Lit
        // evaluation. env off, single bounce, diffuse-only BSDF.
        var t1Point = new Vector3(1.5f, 0, -1.8f);
        sgCompute.Enabled = false;
        var s = ProductionSettings();
        s.envColor = Color.black;
        s.maxBounces = 1;
        s.linearOutput = true;
        s.debugFlags = 1; // diffuse only
        Tracer.Settings = s;
        Tracer.RequestReset();

        for (var i = 0; i < TestFrames; i++) yield return null;

        {
            var albedo = FloorAlbedo(t1Point);
            var (toLight, lightColor) = DirLight();
            var cos = Vector3.Dot(Vector3.up, toLight);
            var expected = (Color)(albedo * lightColor) * (cos / Mathf.PI);
            var measured = ReadResultAverage(WorldToResultPixel(t1Point), 2);
            CheckClose("T1 direct lighting (floor)", measured, expected, 0.03f);
        }

        // T2: furnace test on a convex Lambertian sphere in a uniform
        // environment, viewed through a virtual camera.
        var env = new Color(0.5f, 0.5f, 0.5f);
        s = ProductionSettings();
        s.envColor = env;
        s.lights = Array.Empty<LightDesc>();
        s.maxBounces = 2;
        s.linearOutput = true;
        s.debugFlags = 1; // diffuse only
        Tracer.Settings = s;
        Tracer.CameraOverride =
          LookAtParams(FurnaceCenter + new Vector3(0, 0, -4), FurnaceCenter);
        Tracer.RequestReset();

        for (var i = 0; i < TestFrames; i++) yield return null;

        {
            var rho = _furnaceMat.GetColor("_BaseColor").linear;
            var expected = (Color)(rho * env);
            var result = Tracer.Result;
            var center = new Vector2Int(result.width / 2, result.height / 2);
            var measured = ReadResultAverage(center, 2);
            CheckClose("T2 furnace (rho=0.5 sphere)", measured, expected, 0.02f);
        }

        // T3: hand-written Unity-compiled material evaluation on the floor.
        _procedural.MaterialIndex = _registry.MaterialIndexOf(_floorMat);
        s = ProductionSettings();
        s.envColor = Color.black;
        s.maxBounces = 1;
        s.linearOutput = true;
        s.debugFlags = 1; // diffuse only
        Tracer.Settings = s;
        Tracer.CameraOverride = null;
        Tracer.RequestReset();

        for (var i = 0; i < TestFrames; i++) yield return null;

        {
            var uv = new Vector2((5 - t1Point.x) / 10, (t1Point.z + 5) / 10);
            var cell = uv * ProcCheckerScale;
            var checker = Mathf.Abs(Mathf.Floor(cell.x) + Mathf.Floor(cell.y))
                          % 2;
            var albedo = Color.Lerp(ProcColorA, ProcColorB, checker);
            var (toLight, lightColor) = DirLight();
            var cos = Vector3.Dot(Vector3.up, toLight);
            var expected = (Color)(albedo * lightColor) * (cos / Mathf.PI);
            var measured = ReadResultAverage(WorldToResultPixel(t1Point), 2);
            CheckClose("T3 Unity-compiled material (floor)", measured,
                       expected, 0.03f);
        }

        _procedural.MaterialIndex = _registry.MaterialIndexOf(_whiteMat);
        sgCompute.Enabled = true;

        // T4: Shader Graph generated compute on the floor, tinted for the
        // test so it only passes when the generated kernel really runs.
        _floorMat.SetColor("_BaseColor", new Color(0.55f, 0.75f, 1.0f));
        s = ProductionSettings();
        s.envColor = Color.black;
        s.maxBounces = 1;
        s.linearOutput = true;
        s.debugFlags = 1; // diffuse only
        Tracer.Settings = s;
        Tracer.RequestReset();

        for (var i = 0; i < TestFrames; i++) yield return null;

        {
            var albedo = FloorAlbedo(t1Point);
            var (toLight, lightColor) = DirLight();
            var cos = Vector3.Dot(Vector3.up, toLight);
            var expected = (Color)(albedo * lightColor) * (cos / Mathf.PI);
            var measured = ReadResultAverage(WorldToResultPixel(t1Point), 2);
            CheckClose("T4 Shader Graph generated material (floor)",
                       measured, expected, 0.03f);
        }

        _floorMat.SetColor("_BaseColor", Color.white);

        // T5: vertex color input path (furnace setup on the color quad).
        s = ProductionSettings();
        s.envColor = env;
        s.lights = Array.Empty<LightDesc>();
        s.maxBounces = 2;
        s.linearOutput = true;
        s.debugFlags = 1; // diffuse only
        Tracer.Settings = s;
        var t5Camera = LookAtParams(VertexColorCenter + new Vector3(0, 0, -4),
                                    VertexColorCenter);
        Tracer.CameraOverride = t5Camera;
        Tracer.RequestReset();

        for (var i = 0; i < TestFrames; i++) yield return null;

        {
            var world = VertexColorCenter +
              new Vector3(1f / 3, -1f / 3, 0) * TestQuadScale;
            var albedo = ((Color)QuadCorners[0] + QuadCorners[1] +
                          QuadCorners[2]) / 3;
            albedo.a = 1;
            var expected = (Color)(albedo * env);
            var measured = ReadResultAverage
              (VirtualCameraPixel(t5Camera, world), 2);
            CheckClose("T5 vertex color input (runtime quad)", measured,
                       expected, 0.03f);
        }

        // T7: keyword variants. Enabling _INVERT_ON on the vertex color
        // quad's material must switch the compute shader to its inverted
        // variant (per-material keyword binding).
        _vcMat.EnableKeyword("_INVERT_ON");
        Tracer.RequestReset();

        for (var i = 0; i < TestFrames; i++) yield return null;

        {
            var world = VertexColorCenter +
              new Vector3(1f / 3, -1f / 3, 0) * TestQuadScale;
            var albedo = ((Color)QuadCorners[0] + QuadCorners[1] +
                          QuadCorners[2]) / 3;
            var inverted = new Color(1 - albedo.r, 1 - albedo.g, 1 - albedo.b);
            var expected = (Color)(inverted * env);
            var measured = ReadResultAverage
              (VirtualCameraPixel(t5Camera, world), 2);
            CheckClose("T7 keyword variant (_INVERT_ON)", measured,
                       expected, 0.03f);
        }

        _vcMat.DisableKeyword("_INVERT_ON");

        // T6: alpha clip pass-through on the cutout quad.
        var t6Camera = LookAtParams(CutoutCenter + new Vector3(0, 0, -4),
                                    CutoutCenter);
        Tracer.CameraOverride = t6Camera;
        Tracer.RequestReset();

        for (var i = 0; i < TestFrames; i++) yield return null;

        {
            var clipped = CutoutCenter +
              new Vector3(-0.5f, -0.5f, 0) * TestQuadScale;
            var measured = ReadResultAverage
              (VirtualCameraPixel(t6Camera, clipped), 2);
            CheckClose("T6a alpha clip (clipped cell = env)", measured,
                       env, 0.03f);

            var solid = CutoutCenter +
              new Vector3(0.5f, -0.5f, 0) * TestQuadScale;
            var albedo = new Color(0.6f, 0.6f, 0.6f); // TestCutout.compute
            var expected = (Color)(albedo * env);
            measured = ReadResultAverage(VirtualCameraPixel(t6Camera, solid), 2);
            CheckClose("T6b alpha clip (solid cell = albedo*env)", measured,
                       expected, 0.03f);
        }

        // T8: punctual lights (URP attenuation conventions) on a flat gray
        // receiver quad. The light sits on the quad's normal axis 3 units
        // away, so at the center: cos = 1, d^2 = 9. Expected radiance =
        // albedo * color * intensity * window(d,range) / d^2.
        var t8Camera = LookAtParams(PunctualCenter + new Vector3(0, 0, -5),
                                    PunctualCenter);
        Color PunctualExpected(Light light)
        {
            var albedo = _punctualMat.GetColor("_BaseColor").linear;
            const float d2 = 9;
            var r2 = light.range * light.range;
            var window = Mathf.Pow
              (Mathf.Clamp01(1 - (d2 / r2) * (d2 / r2)), 2);
            var c = light.color.linear * light.intensity * (window / d2);
            return albedo * c;
        }

        _light.enabled = false;
        _pointLight.enabled = true;
        s = ProductionSettings(); // picks up the point light only
        s.envColor = Color.black;
        s.maxBounces = 1;
        s.linearOutput = true;
        s.debugFlags = 1; // diffuse only
        Tracer.Settings = s;
        Tracer.CameraOverride = t8Camera;
        Tracer.RequestReset();

        for (var i = 0; i < TestFrames; i++) yield return null;

        {
            var measured = ReadResultAverage
              (VirtualCameraPixel(t8Camera, PunctualCenter), 2);
            CheckClose("T8a point light (quad center)", measured,
                       PunctualExpected(_pointLight), 0.03f);
        }

        _pointLight.enabled = false;
        _spotLight.enabled = true;
        s = ProductionSettings(); // picks up the spot light only
        s.envColor = Color.black;
        s.maxBounces = 1;
        s.linearOutput = true;
        s.debugFlags = 1;
        Tracer.Settings = s;
        Tracer.RequestReset();

        for (var i = 0; i < TestFrames; i++) yield return null;

        {
            // On the spot axis the angle attenuation saturates to 1.
            var measured = ReadResultAverage
              (VirtualCameraPixel(t8Camera, PunctualCenter), 2);
            CheckClose("T8b spot light (on axis)", measured,
                       PunctualExpected(_spotLight), 0.03f);
        }

        _spotLight.enabled = false;

        // T9: alpha-tested shadows. The half-alpha occluder sits between
        // the point light and the receiver; swapping _Cutoff around the
        // 0.5 base map alpha toggles it between shadow-transparent and
        // shadow-opaque (native Lit alpha evaluation on the shadow path).
        // The camera is elevated so primary rays miss the small occluder.
        var t9Camera = LookAtParams(ShadowCenter + new Vector3(0, 2, -8),
                                    ShadowCenter);
        _shadowLight.enabled = true;

        Color ShadowExpected()
        {
            var albedo = _shadowRecvMat.GetColor("_BaseColor").linear;
            const float d2 = 9;
            var r2 = _shadowLight.range * _shadowLight.range;
            var window = Mathf.Pow
              (Mathf.Clamp01(1 - (d2 / r2) * (d2 / r2)), 2);
            var c = _shadowLight.color.linear * _shadowLight.intensity *
                    (window / d2);
            return albedo * c;
        }

        // Phase A: cutoff above the base map alpha -> shadow rays pass.
        _shadowOccMat.SetFloat("_Cutoff", 0.8f);
        _registry.RefreshMaterials();
        s = ProductionSettings(); // shadow point light only
        s.envColor = Color.black;
        s.maxBounces = 1;
        s.linearOutput = true;
        s.debugFlags = 1;
        Tracer.Settings = s;
        Tracer.CameraOverride = t9Camera;
        Tracer.RequestReset();

        for (var i = 0; i < TestFrames; i++) yield return null;

        {
            var measured = ReadResultAverage
              (VirtualCameraPixel(t9Camera, ShadowCenter), 2);
            CheckClose("T9a alpha-tested shadow (transparent occluder)",
                       measured, ShadowExpected(), 0.03f);
        }

        // Phase B: cutoff below the base map alpha -> shadow rays blocked.
        _shadowOccMat.SetFloat("_Cutoff", 0.3f);
        _registry.RefreshMaterials();
        Tracer.RequestReset();

        for (var i = 0; i < TestFrames; i++) yield return null;

        {
            var measured = ReadResultAverage
              (VirtualCameraPixel(t9Camera, ShadowCenter), 2);
            CheckClose("T9b alpha-tested shadow (opaque occluder)",
                       measured, Color.black, 0.03f);
        }

        _shadowOccMat.SetFloat("_Cutoff", 0.8f);
        _registry.RefreshMaterials();
        _shadowLight.enabled = false;
        _light.enabled = true;

        // Production: progressive path tracing of the visible scene.
        Tracer.Settings = ProductionSettings();
        Tracer.CameraOverride = null;
        Tracer.RequestReset();
        _tracedFrames = 0;

        while (_tracedFrames < SaveFrame) yield return null;

        Log($"Render thread events executed: {MetalRT_GetEventFrameCount()} " +
            $"(tracer frames issued: {Tracer.FrameIndex})");
        SaveResultImage();
    }

    Vector2Int WorldToResultPixel(Vector3 worldPos)
    {
        var sp = _camera.WorldToScreenPoint(worldPos);
        var p = new Vector2Int(Mathf.RoundToInt(sp.x), Mathf.RoundToInt(sp.y));
        var result = Tracer.Result;
        if (sp.z <= 0 || p.x < 2 || p.y < 2 ||
            p.x >= result.width - 2 || p.y >= result.height - 2)
            Log($"WARNING: test point {worldPos} projects off screen ({p})");
        return p;
    }

    static Vector2Int VirtualCameraPixel(in TraceParams p, Vector3 world)
    {
        Vector3 origin = p.originTan;
        Vector3 fwd = p.forward, right = p.rightAspect, up = p.up;
        var tan = p.originTan.w;
        var aspect = p.rightAspect.w;
        var d = world - origin;
        var z = Vector3.Dot(d, fwd);
        var ndcX = Vector3.Dot(d, right) / z / (tan * aspect);
        var ndcY = Vector3.Dot(d, up) / z / tan;
        return new Vector2Int(
            Mathf.RoundToInt((ndcX * 0.5f + 0.5f) * p.width),
            Mathf.RoundToInt((ndcY * 0.5f + 0.5f) * p.height));
    }

    TraceParams LookAtParams(Vector3 position, Vector3 target)
    {
        var fwd = (target - position).normalized;
        var right = Vector3.Cross(Vector3.up, fwd).normalized;
        var up = Vector3.Cross(fwd, right);
        var result = Tracer.Result;
        var (w, h) = (result.width, result.height);
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

    Color ReadResultAverage(Vector2Int center, int radius)
    {
        var n = radius * 2 + 1;
        var prev = RenderTexture.active;
        RenderTexture.active = Tracer.Result;
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
        var result = Tracer.Result;
        var (w, h) = (result.width, result.height);
        var prev = RenderTexture.active;
        RenderTexture.active = result;
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
