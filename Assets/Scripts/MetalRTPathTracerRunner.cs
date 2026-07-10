using UnityEngine;
using UnityEngine.Rendering;
using static MetalRTTest.MetalRTPlugin;

namespace MetalRTTest {

// Minimal scene component that drives the Metal RT path tracer without the
// test harness: put this in a scene (see Assets/Scenes/Sample.unity), give
// a camera the path tracer renderer (renderer index 1), and press play.
// The scene's MeshRenderers and lights are registered automatically.
public sealed class MetalRTPathTracerRunner : MonoBehaviour
{
    [field:SerializeField, Range(1, 8)]
    public int MaxBounces { get; set; } = 5;

    [field:SerializeField]
    public float Exposure { get; set; } = 1;

    [field:SerializeField]
    public bool ShowLabels { get; set; } = true;

    MetalRTSceneRegistry _registry;

    static MetalRTPathTracer Tracer => MetalRTPathTracer.Instance;

    void Start()
    {
        if (MetalRT_DeviceSupportsRaytracing() != 1)
        {
            Debug.LogError("[MetalRT] This device does not support Metal " +
                           "hardware ray tracing.");
            enabled = false;
            return;
        }

        _registry = new MetalRTSceneRegistry();
        if (!_registry.Build(Tracer))
        {
            enabled = false;
            return;
        }

        Tracer.Configure(_registry.MakeDescs);
        ApplySettings();
    }

    void Update() => ApplySettings();

    void OnDestroy()
    {
        Tracer.Dispose();
        MetalRT_Dispose();
    }

    void OnGUI()
    {
        if (!ShowLabels || !Tracer.IsConfigured) return;
        GUI.Label(new Rect(10, 10, 300, 20), "URP Raster");
        GUI.Label(new Rect(Screen.width / 2 + 10, 10, 300, 20),
                  "Metal RT Path Tracer");
    }

    void ApplySettings() => Tracer.Settings = new FrameSettings
    {
        envColor = RenderSettings.ambientLight.linear,
        lights = MetalRTSceneRegistry.CollectLights(),
        maxBounces = (uint)MaxBounces,
        exposure = Exposure
    };
}

} // namespace MetalRTTest
