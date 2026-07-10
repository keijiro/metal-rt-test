using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;

namespace UrpMetalPathTracer {

// URP renderer feature that replaces the camera's color output with the
// Metal RT path traced image. The pass records the native phase events and
// the material evaluation compute dispatches through a RenderGraph unsafe
// pass (the same escape hatch used by DLSS/FSR2-style native integrations),
// then blits the progressive result over the camera color target.
public sealed class MetalRTPathTracerFeature : ScriptableRendererFeature
{
    MetalRTPathTracerPass _pass;

    public override void Create()
      => _pass = new MetalRTPathTracerPass
           { renderPassEvent = RenderPassEvent.AfterRenderingPostProcessing };

    public override void AddRenderPasses(ScriptableRenderer renderer,
                                         ref RenderingData renderingData)
    {
        if (MetalRTPathTracer.Instance.IsConfigured)
            renderer.EnqueuePass(_pass);
    }
}

sealed class MetalRTPathTracerPass : ScriptableRenderPass
{
    class PassData
    {
        public Camera Camera;
        public TextureHandle Color;
    }

    public override void RecordRenderGraph(RenderGraph renderGraph,
                                           ContextContainer frameData)
    {
        var resourceData = frameData.Get<UniversalResourceData>();
        var cameraData = frameData.Get<UniversalCameraData>();

        using var builder = renderGraph.AddUnsafePass<PassData>
          ("MetalRT Path Tracer", out var data);
        data.Camera = cameraData.camera;
        data.Color = resourceData.activeColorTexture;
        builder.UseTexture(data.Color, AccessFlags.Write);
        builder.AllowPassCulling(false);
        builder.SetRenderFunc((PassData d, UnsafeGraphContext ctx) =>
        {
            var cmd = CommandBufferHelpers.GetNativeCommandBuffer(ctx.cmd);
            var tracer = MetalRTPathTracer.Instance;
            tracer.Record(cmd, d.Camera);
            if (tracer.ResultHandle == null) return;
            cmd.SetRenderTarget(d.Color);
            Blitter.BlitTexture(cmd, tracer.ResultHandle,
                                new Vector4(1, 1, 0, 0), 0, false);
        });
    }
}

} // namespace UrpMetalPathTracer
