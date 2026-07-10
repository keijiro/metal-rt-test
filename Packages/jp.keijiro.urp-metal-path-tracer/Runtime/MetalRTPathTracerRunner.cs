using UnityEngine;
using UnityEngine.Rendering;
using static UrpMetalPathTracer.MetalRTPlugin;

namespace UrpMetalPathTracer {

// Minimal scene component that drives the Metal RT path tracer without the
// test harness: put this in a scene (see Assets/Scenes/Sample.unity), give
// a camera the path tracer renderer (renderer index 1), and press play.
// The scene's MeshRenderers and lights are registered automatically and
// kept in sync, so objects can be moved, added, or removed at runtime
// (accumulation restarts on any change).
//
// The component also runs in edit mode. Since the editor only redraws on
// demand, accumulation normally progresses only while the editor repaints
// (e.g., while interacting); enable ContinuousEditRefresh to keep the
// player loop ticking so the image converges in edit mode too.
[ExecuteAlways]
public sealed class MetalRTPathTracerRunner : MonoBehaviour
{
    [field:SerializeField, Range(1, 8)]
    public int MaxBounces { get; set; } = 5;

    [field:SerializeField]
    public float Exposure { get; set; } = 1;

    [field:SerializeField]
    public bool ShowLabels { get; set; } = true;

    [field:SerializeField]
    public bool ContinuousEditRefresh { get; set; } = false;

    MetalRTSceneRegistry _registry;

    static MetalRTPathTracer Tracer => MetalRTPathTracer.Instance;

    void OnEnable()
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
            _registry = null;
            enabled = false;
            return;
        }

        Tracer.Configure(_registry.MakeDescs);
        ApplySettings();

#if UNITY_EDITOR
        UnityEditor.EditorApplication.update += OnEditorUpdate;
#endif
    }

    void OnDisable()
    {
#if UNITY_EDITOR
        UnityEditor.EditorApplication.update -= OnEditorUpdate;
#endif
        if (_registry == null) return;
        _registry = null;
        Tracer.Dispose();
        MetalRT_Dispose();
    }

    void Update()
    {
        if (_registry == null) return;
        _registry.Sync(Tracer); // pick up added/removed/re-materialed objects
        ApplySettings();
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

#if UNITY_EDITOR
    // Keeps the player loop (and therefore the game view and the path
    // tracer) ticking in edit mode when continuous refresh is requested.
    void OnEditorUpdate()
    {
        if (!Application.isPlaying && ContinuousEditRefresh)
            UnityEditor.EditorApplication.QueuePlayerLoopUpdate();
    }
#endif
}

} // namespace UrpMetalPathTracer
