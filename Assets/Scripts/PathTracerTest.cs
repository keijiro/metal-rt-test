using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using static MetalRTTest.MetalRTPlugin;

namespace MetalRTTest {

// Test harness for the Metal RT path tracer. Builds a static URP scene at
// runtime, registers its meshes and materials with the native plugin, and
// runs analytic verification passes. Rendering happens through URP itself:
// the left camera rasterizes the scene normally, while the right camera
// uses a renderer with MetalRTPathTracerFeature, which replaces its output
// with the path traced image.
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
        (new Vector3(0, 0.4f, 0), Vector3.right, true, 0.6f, 1),
        (new Vector3(0, 5, 0), Vector3.down, true, 5.0f, 0),
        (new Vector3(2.2f, 5, 0.5f), Vector3.down, true, 3.8f, 2),
        (new Vector3(0, 1, -6), Vector3.back, false, 0, 0),
        (new Vector3(-5, 0.4f, 0), Vector3.right, true, 3.6f, 1),
    };

    const float ProbeTolerance = 0.02f;

    // Furnace test sphere, far above the scene (only the analytic tests'
    // virtual camera looks at it).
    static readonly Vector3 FurnaceCenter = new Vector3(0, 50, 0);
    const float FurnaceScale = 2; // base radius 0.5 -> radius 1

    const int TestFrames = 32;   // accumulation frames per analytic test
    const int SaveFrame = 300;   // progressive frame to save the PNG at
    const uint ProductionBounces = 5;

    // Procedural test material parameters (linear colors)
    static readonly Color ProcColorA = new Color(0.9f, 0.25f, 0.1f);
    static readonly Color ProcColorB = new Color(0.1f, 0.3f, 0.9f);
    const float ProcCheckerScale = 6;
    const int ProceduralTargetMaterial = 3; // white torus

    // Private members

    Camera _camera;   // left half: URP raster reference
    Camera _ptCamera; // right half: URP camera with the path tracer feature
    Light _light;
    (int mesh, int material, Transform transform)[] _instances;
    Mesh[] _meshes = new Mesh[0];
    Material[] _materials;
    MetalRTPathTracer.MaterialCompute _procedural;
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

        if (!SetUpScene()) return;
        if (!SetUpMaterials()) return;
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

    // Test scene construction

    bool SetUpScene()
    {
        var plane = LoadAndRegisterMesh("Plane");
        var torus = LoadAndRegisterMesh("Torus");
        var sphere = LoadAndRegisterMesh("Sphere");
        if (plane < 0 || torus < 0 || sphere < 0) return false;

        // The floor uses the test Shader Graph: URP rasterizes it with the
        // graph's own shader while the path tracer evaluates the compute
        // shader generated from the same graph, so both views must match.
        var floorMat = MakeShaderGraphMaterial();
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
        return true;
    }

    FrameSettings ProductionSettings() => new FrameSettings
    {
        envColor = RenderSettings.ambientLight.linear,
        lightDir = _light.transform.forward,
        // Unity's punctual light convention folds 1/pi into the Lambert
        // term, so premultiply by pi for the physically normalized BRDF.
        lightColor = _light.color.linear * (_light.intensity * Mathf.PI),
        maxBounces = ProductionBounces,
        exposure = 1
    };

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

        var normalOffset = AttributeOffset(mesh, VertexAttribute.Normal,
                                           posStream, 3);
        var tangentOffset = AttributeOffset(mesh, VertexAttribute.Tangent,
                                            posStream, 4);
        var uvOffset = AttributeOffset(mesh, VertexAttribute.TexCoord0,
                                       posStream, 2);

        var indexCount = mesh.GetIndexCount(0);
        var indexStart = mesh.GetIndexStart(0);
        var is16Bit = mesh.indexFormat == IndexFormat.UInt16;
        var indexSize = is16Bit ? 2u : 4u;

        var ret = MetalRT_AddMesh
          (mesh.GetNativeVertexBufferPtr(posStream), (uint)stride,
           (uint)posOffset, normalOffset, tangentOffset, uvOffset,
           mesh.GetNativeIndexBufferPtr(),
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
            var tex = mat.HasProperty("_BaseMap") ?
              mat.GetTexture("_BaseMap") as Texture2D : null;
            var emission = mat.IsKeywordEnabled("_EMISSION") &&
                           mat.HasProperty("_EmissionColor") ?
              mat.GetColor("_EmissionColor") : Color.black;
            descs[i] = new MaterialDesc
            {
                baseColor = mat.HasProperty("_BaseColor") ?
                  mat.GetColor("_BaseColor").linear : Color.white,
                emission = emission, // HDR colors are already linear
                baseMapST = mat.HasProperty("_BaseMap_ST") ?
                  mat.GetVector("_BaseMap_ST") : new Vector4(1, 1, 0, 0),
                metallic = mat.HasProperty("_Metallic") ?
                  mat.GetFloat("_Metallic") : 0,
                smoothness = mat.HasProperty("_Smoothness") ?
                  mat.GetFloat("_Smoothness") : 0.5f,
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

    // Tracer configuration (buffers, material evaluation computes)

    bool SetUpTracer()
    {
        Tracer.Configure(MakeInstanceDescs);
        Tracer.Settings = ProductionSettings();
        if (!Tracer.EnsureResources(_ptCamera.pixelWidth,
                                    _ptCamera.pixelHeight))
        {
            Log("FAIL: tracer resource setup");
            return false;
        }
        Log($"Tracer configured ({_ptCamera.pixelWidth}x" +
            $"{_ptCamera.pixelHeight}); driven by the URP renderer feature");

        var procCs = Resources.Load<ComputeShader>("TestProcedural");
        _procedural = new MetalRTPathTracer.MaterialCompute
        {
            Shader = procCs,
            Kernel = procCs.FindKernel("EvaluateMaterial"),
            MaterialIndex = ProceduralTargetMaterial,
            Bind = cb =>
            {
                cb.SetComputeVectorParam(procCs, "_ColorA", ProcColorA);
                cb.SetComputeVectorParam(procCs, "_ColorB", ProcColorB);
                cb.SetComputeFloatParam(procCs, "_CheckerScale",
                                        ProcCheckerScale);
            }
        };
        Tracer.MaterialComputes.Add(_procedural);
        Log("Procedural material compute registered " +
            $"(material {ProceduralTargetMaterial}, Unity-compiled kernel)");

        var sgCompute = Resources.Load<ComputeShader>("TestGraphGen");
        if (sgCompute == null)
        {
            Log("FAIL: TestGraphGen.compute not found " +
                "(run MetalRT/Generate Compute From Test Graph)");
            return false;
        }
        Tracer.MaterialComputes.Add
          (MakeShaderGraphCompute(sgCompute, _materials[0], 0));
        Log("Shader Graph material compute registered (material 0, " +
            "generated from TestGraph.shadergraph)");
        return true;
    }

    // Wraps a generated Shader Graph compute shader with a binder that
    // feeds it the source material's properties.
    MetalRTPathTracer.MaterialCompute MakeShaderGraphCompute
      (ComputeShader cs, Material mat, int materialIndex)
    {
        var kernel = cs.FindKernel("EvaluateMaterial");
        var shader = mat.shader;
        var floats = new System.Collections.Generic.List<string>();
        var colors = new System.Collections.Generic.List<(string, bool)>();
        var vectors = new System.Collections.Generic.List<string>();
        var textures = new System.Collections.Generic.List<(string, string)>();

        for (var i = 0; i < shader.GetPropertyCount(); i++)
        {
            var name = shader.GetPropertyName(i);
            var flags = shader.GetPropertyFlags(i);
            switch (shader.GetPropertyType(i))
            {
                case ShaderPropertyType.Float:
                case ShaderPropertyType.Range:
                    floats.Add(name);
                    break;
                case ShaderPropertyType.Color:
                    colors.Add((name, flags.HasFlag(ShaderPropertyFlags.HDR)));
                    break;
                case ShaderPropertyType.Vector:
                    vectors.Add(name);
                    break;
                case ShaderPropertyType.Texture:
                    textures.Add((name,
                                  shader.GetPropertyTextureDefaultName(i)));
                    break;
            }
        }

        return new MetalRTPathTracer.MaterialCompute
        {
            Shader = cs,
            Kernel = kernel,
            MaterialIndex = materialIndex,
            Bind = cb =>
            {
                foreach (var n in floats)
                    cb.SetComputeFloatParam(cs, n, mat.GetFloat(n));
                foreach (var (n, hdr) in colors)
                    cb.SetComputeVectorParam
                      (cs, n, hdr ? mat.GetColor(n) : mat.GetColor(n).linear);
                foreach (var n in vectors)
                    cb.SetComputeVectorParam(cs, n, mat.GetVector(n));
                foreach (var (n, def) in textures)
                {
                    var t = mat.GetTexture(n) ?? DefaultTexture(def);
                    cb.SetComputeTextureParam(cs, kernel, n, t);
                    cb.SetComputeVectorParam
                      (cs, n + "_TexelSize",
                       new Vector4(1f / t.width, 1f / t.height,
                                   t.width, t.height));
                    if (mat.HasProperty(n + "_ST"))
                        cb.SetComputeVectorParam
                          (cs, n + "_ST", mat.GetVector(n + "_ST"));
                }
                cb.SetComputeVectorParam
                  (cs, "_TimeParams",
                   new Vector4(Time.time, Mathf.Sin(Time.time),
                               Mathf.Cos(Time.time), 0));
            }
        };
    }

    static Texture DefaultTexture(string name) => name switch
    {
        "black" => Texture2D.blackTexture,
        "bump" => Texture2D.normalTexture,
        "gray" or "grey" => Texture2D.grayTexture,
        _ => Texture2D.whiteTexture
    };

    // Analytic radiance tests (T1 direct lighting, T2 furnace, T3/T4
    // Unity-compiled material evaluation), then the production render.

    IEnumerator RunTestSequence()
    {
        var sgCompute = Tracer.MaterialComputes.Count > 1 ?
          Tracer.MaterialComputes[^1] : null;

        // Expected linear floor albedo at a world point: checker base map
        // sample times the base color tint (cell interiors only).
        Color FloorAlbedo(Vector3 p)
        {
            var uv = new Vector2((5 - p.x) / 10, (p.z + 5) / 10);
            var cell = uv * 8; // checker texture cells
            var odd = (Mathf.FloorToInt(cell.x) + Mathf.FloorToInt(cell.y))
                      % 2 == 1;
            var texel = Mathf.GammaToLinearSpace((odd ? 230 : 100) / 255f);
            return texel * _materials[0].GetColor("_BaseColor").linear;
        }

        // T1: direct lighting on the floor through the native URP Lit
        // evaluation. env off, single bounce, diffuse-only BSDF.
        var t1Point = new Vector3(1.5f, 0, -1.8f);
        if (sgCompute != null) sgCompute.Enabled = false;
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
            var cos = Vector3.Dot(Vector3.up, -s.lightDir);
            var expected = (Color)(albedo * s.lightColor) * (cos / Mathf.PI);
            var measured = ReadResultAverage(WorldToResultPixel(t1Point), 2);
            CheckClose("T1 direct lighting (floor)", measured, expected, 0.03f);
        }

        // T2: furnace test on a convex Lambertian sphere in a uniform
        // environment, viewed through a virtual camera.
        var env = new Color(0.5f, 0.5f, 0.5f);
        s = ProductionSettings();
        s.envColor = env;
        s.lightColor = Color.black;
        s.maxBounces = 2;
        s.linearOutput = true;
        s.debugFlags = 1; // diffuse only
        Tracer.Settings = s;
        Tracer.CameraOverride =
          LookAtParams(FurnaceCenter + new Vector3(0, 0, -4), FurnaceCenter);
        Tracer.RequestReset();

        for (var i = 0; i < TestFrames; i++) yield return null;

        {
            var rho = _materials[5].GetColor("_BaseColor").linear;
            var expected = (Color)(rho * env);
            var result = Tracer.Result;
            var center = new Vector2Int(result.width / 2, result.height / 2);
            var measured = ReadResultAverage(center, 2);
            CheckClose("T2 furnace (rho=0.5 sphere)", measured, expected, 0.02f);
        }

        // T3: hand-written Unity-compiled material evaluation on the floor.
        _procedural.MaterialIndex = 0; // floor (sg compute stays disabled)
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
            var cos = Vector3.Dot(Vector3.up, -s.lightDir);
            var expected = (Color)(albedo * s.lightColor) * (cos / Mathf.PI);
            var measured = ReadResultAverage(WorldToResultPixel(t1Point), 2);
            CheckClose("T3 Unity-compiled material (floor)", measured,
                       expected, 0.03f);
        }

        _procedural.MaterialIndex = ProceduralTargetMaterial;
        if (sgCompute != null) sgCompute.Enabled = true;

        // T4: Shader Graph generated compute on the floor. The base color
        // is tinted during the test: the generated compute reads material
        // properties live, while the native Lit fallback keeps its setup
        // snapshot (white) — so the test only passes when the generated
        // kernel really runs.
        if (sgCompute != null)
        {
            _materials[0].SetColor("_BaseColor",
                                   new Color(0.55f, 0.75f, 1.0f));
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
                var cos = Vector3.Dot(Vector3.up, -s.lightDir);
                var expected = (Color)(albedo * s.lightColor) *
                               (cos / Mathf.PI);
                var measured = ReadResultAverage(WorldToResultPixel(t1Point), 2);
                CheckClose("T4 Shader Graph generated material (floor)",
                           measured, expected, 0.03f);
            }

            _materials[0].SetColor("_BaseColor", Color.white);
        }

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
        // The left (reference) camera shares pose/fov/aspect with the path
        // traced camera, so its projection maps 1:1 onto the result texture.
        var sp = _camera.WorldToScreenPoint(worldPos);
        var p = new Vector2Int(Mathf.RoundToInt(sp.x), Mathf.RoundToInt(sp.y));
        var result = Tracer.Result;
        if (sp.z <= 0 || p.x < 2 || p.y < 2 ||
            p.x >= result.width - 2 || p.y >= result.height - 2)
            Log($"WARNING: test point {worldPos} projects off screen ({p})");
        return p;
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

    // Reads back a small region of the (linear output mode) result texture
    // and returns the average color.
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
