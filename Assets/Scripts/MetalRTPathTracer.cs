using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;
using static MetalRTTest.MetalRTPlugin;

namespace MetalRTTest {

// Runtime core of the Metal RT path tracer. Owns the GPU-facing state
// (result texture, wavefront buffers, event data blobs, material evaluation
// compute shaders) and records the per-frame phase pipeline into a command
// buffer. The URP renderer feature (MetalRTPathTracerFeature) calls Record
// once per camera render; the scene side registers meshes/materials/
// instances through the plugin and configures this class.
public sealed class MetalRTPathTracer
{
    public static MetalRTPathTracer Instance { get; private set; } = new();

    // A material evaluated by a Unity-compiled compute shader instead of
    // the native default URP Lit evaluation (the Shader Graph path).
    public sealed class MaterialCompute
    {
        public ComputeShader Shader;
        public int Kernel;
        public int MaterialIndex;
        public bool Enabled = true;
        public Action<CommandBuffer> Bind;
    }

    // Public state

    public FrameSettings Settings;
    public TraceParams? CameraOverride; // analytic tests use a virtual camera
    public List<MaterialCompute> MaterialComputes { get; } = new();
    public RenderTexture Result => _result;
    public bool IsConfigured => _instancesProvider != null;
    public uint FrameIndex => _frameIndex;

    // Private members

    Func<InstanceDesc[]> _instancesProvider;
    RenderTexture _result;
    RTHandle _resultHandle;
    IntPtr _resultPtr;
    GraphicsBuffer _hitBuffer, _attrBuffer, _surfBuffer;
    IntPtr _eventFunc;
    IntPtr[] _eventData;
    int _eventSlot;
    uint _frameIndex;
    bool _resetQueued = true;
    Matrix4x4 _lastCamera;
    float _lastFov;

    const int EventRingSize = 4;

    public RTHandle ResultHandle => _resultHandle;

    // Setup / teardown (called by the scene harness)

    public void Configure(Func<InstanceDesc[]> instancesProvider)
    {
        _instancesProvider = instancesProvider;
        _eventFunc = MetalRT_GetRenderEventFunc();

        if (_eventData == null)
        {
            _eventData = new IntPtr[EventRingSize];
            for (var i = 0; i < EventRingSize; i++)
                _eventData[i] = Marshal.AllocHGlobal(EventDataSize);
        }
    }

    public void RequestReset() => _resetQueued = true;

    public void Dispose()
    {
        _instancesProvider = null;
        _resultHandle?.Release();
        _resultHandle = null;
        if (_result != null) _result.Release();
        _result = null;
        _hitBuffer?.Dispose();
        _attrBuffer?.Dispose();
        _surfBuffer?.Dispose();
        _hitBuffer = _attrBuffer = _surfBuffer = null;
        if (_eventData != null)
            foreach (var ptr in _eventData) Marshal.FreeHGlobal(ptr);
        _eventData = null;
        MaterialComputes.Clear();
        Instance = new MetalRTPathTracer();
    }

    // Creates (or resizes) the result texture and the wavefront buffers
    // shared with the native plugin.
    public bool EnsureResources(int width, int height)
    {
        if (_result != null &&
            _result.width == width && _result.height == height) return true;
        if (width <= 0 || height <= 0) return false;

        _resultHandle?.Release();
        if (_result != null) _result.Release();
        _hitBuffer?.Dispose();
        _attrBuffer?.Dispose();
        _surfBuffer?.Dispose();

        _result = new RenderTexture(width, height, 0,
                                    RenderTextureFormat.ARGBFloat)
          { enableRandomWrite = true };
        _result.Create();
        _resultPtr = _result.GetNativeTexturePtr();
        _resultHandle = RTHandles.Alloc(_result);

        var pixels = width * height;
        _hitBuffer = new GraphicsBuffer
          (GraphicsBuffer.Target.Structured, pixels, HitRecordStride);
        _attrBuffer = new GraphicsBuffer
          (GraphicsBuffer.Target.Structured, pixels, HitAttributesStride);
        _surfBuffer = new GraphicsBuffer
          (GraphicsBuffer.Target.Structured, pixels, SurfaceRecordStride);
        var ret = MetalRT_SetSharedBuffers(_hitBuffer.GetNativeBufferPtr(),
                                           _attrBuffer.GetNativeBufferPtr(),
                                           _surfBuffer.GetNativeBufferPtr());
        if (ret != 0)
        {
            Debug.LogError($"[MetalRT] Shared buffer registration failed: " +
                           LastError);
            return false;
        }

        _resetQueued = true;
        return true;
    }

    // Records one frame of the path tracing pipeline (TLAS rebuild, RayGen,
    // per-bounce Intersect / material evaluation / Shade, Resolve) into the
    // given command buffer. Called from the render pass.
    public void Record(CommandBuffer cmd, Camera camera)
    {
        if (!IsConfigured) return;
        if (!EnsureResources(camera.pixelWidth, camera.pixelHeight)) return;

        // Restart accumulation when the camera moves (unless an analytic
        // test drives a virtual camera and manages resets itself).
        if (CameraOverride == null)
        {
            var m = camera.transform.localToWorldMatrix;
            if (m != _lastCamera || camera.fieldOfView != _lastFov)
                _resetQueued = true;
            _lastCamera = m;
            _lastFov = camera.fieldOfView;
        }

        var p = CameraOverride ?? CameraParams(camera);
        var s = Settings;
        s.reset = _resetQueued;
        _resetQueued = false;

        _eventSlot = (_eventSlot + 1) % EventRingSize;
        var blob = _eventData[_eventSlot];
        WriteEventData(blob, p, _resultPtr, _frameIndex++, s,
                       _instancesProvider());

        var (w, h) = (_result.width, _result.height);
        var (gx, gy) = ((w + 7) / 8, (h + 7) / 8);

        cmd.IssuePluginEventAndData(_eventFunc, PhaseBegin, blob);
        for (var b = 0; b < s.maxBounces; b++)
        {
            cmd.IssuePluginEventAndData(_eventFunc, PhaseIntersect | b << 8,
                                        blob);
            foreach (var mc in MaterialComputes)
            {
                if (!mc.Enabled) continue;
                cmd.SetComputeBufferParam(mc.Shader, mc.Kernel,
                                          "_Attributes", _attrBuffer);
                cmd.SetComputeBufferParam(mc.Shader, mc.Kernel,
                                          "_Surfaces", _surfBuffer);
                cmd.SetComputeIntParam(mc.Shader, "_Width", w);
                cmd.SetComputeIntParam(mc.Shader, "_Height", h);
                cmd.SetComputeIntParam(mc.Shader, "_MaterialIndex",
                                       mc.MaterialIndex);
                mc.Bind?.Invoke(cmd);
                cmd.DispatchCompute(mc.Shader, mc.Kernel, gx, gy, 1);
            }
            cmd.IssuePluginEventAndData(_eventFunc, PhaseShade | b << 8, blob);
        }
        cmd.IssuePluginEventAndData(_eventFunc, PhaseResolve, blob);
    }

    TraceParams CameraParams(Camera camera)
    {
        var ct = camera.transform;
        var tanFov = Mathf.Tan(camera.fieldOfView * Mathf.Deg2Rad / 2);
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

    static Vector4 V4(Vector3 v, float w) => new Vector4(v.x, v.y, v.z, w);
}

} // namespace MetalRTTest
