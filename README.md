# metal-rt-test

A verification project for using Mac hardware ray tracing (MetalRT) with Unity.

![comparison](Images/comparison.png)

## Purpose

This project verifies that hardware ray tracing works on macOS (Apple Silicon)
with meshes that live in Unity — specifically, that an acceleration structure
can be built from a non-readable Unity `Mesh` (`isReadable == false`, i.e. no
CPU copy) and that ray intersection tests against it produce correct results.
This is a data-level test: no materials or lighting, just an image output that
visually confirms intersections are correct.

The end goal of this line of work is cooperation with URP rendering; this
project covers only the fundamental feasibility check.

## Background

As of Unity 6.3 (6000.3), Unity's native ray tracing APIs
(`RayTracingAccelerationStructure`, `RayTracingShader`, inline `RayQuery`) are
not supported on Metal — `SystemInfo.supportsRayTracing` returns `false`, and
Metal support is a long-term item with no announced release. The
`UnifiedRayTracing` API falls back to a compute shader implementation on Metal,
which is not hardware ray tracing.

Therefore, the only way to use hardware ray tracing on Mac is a native plugin
that calls the Metal ray tracing APIs (`MTLAccelerationStructure` +
`metal::raytracing::intersector`) directly. That plugin is the core of this
project.

## How it works

- `NativePlugin/MetalRTPlugin.mm` — The native plugin. It obtains Unity's
  `MTLDevice` through `IUnityGraphicsMetalV1`, builds per-mesh primitive
  acceleration structures (BLAS) directly from the mesh GPU buffers, combines
  them into an instance acceleration structure (TLAS) with per-instance
  transforms, and dispatches compute kernels (compiled at runtime from
  embedded MSL source) that trace world-space rays with
  `raytracing::intersector<triangle_data, instancing>`. Per-instance
  vertex/index buffers for hit shading are accessed bindlessly through GPU
  addresses (`MTLBuffer.gpuAddress`, Metal 3). One-time setup (BLAS builds)
  and the probe tests run synchronously on a private command queue; the
  per-frame TLAS rebuild and full-frame trace are encoded directly into
  Unity's own Metal command buffer on the render thread — a plugin render
  event ends Unity's current command encoder (`EndCurrentCommandEncoder`),
  adds an acceleration structure build encoder and a compute encoder to
  `CurrentCommandBuffer`, and returns. The work is GPU-ordered with Unity's
  rendering and the CPU never blocks.
- `Assets/Scripts/MetalRayTracingTest.cs` — The test driver. It requires no
  scene setup (it bootstraps itself via `RuntimeInitializeOnLoadMethod`) and:
  - Loads the torus and sphere meshes, which are non-readable by default as
    imported model assets, and passes their native vertex/index buffer
    pointers (`Mesh.GetNativeVertexBufferPtr` / `GetNativeIndexBufferPtr`) to
    the plugin to build one BLAS per mesh — no CPU-side mesh data is ever
    touched.
  - Places three instances (two tori sharing one BLAS, one sphere) and
    rebuilds the TLAS from their `Transform`s every frame, so animated
    transforms are picked up.
  - Runs world-space probe rays with analytically derived expectations
    (hit distances, hit instance indices, misses through the torus hole) and
    logs PASS/FAIL.
  - Traces a full frame from the scene camera every frame, dispatched on the
    render thread through `CommandBuffer.IssuePluginEventAndData` (per-frame
    parameters and instance transforms are passed in a small ring of
    unmanaged blobs) and written by the native plugin directly into a Unity
    `RenderTexture` (created with `enableRandomWrite` and passed as
    `GetNativeTexturePtr`). The result is shown next to a rasterized
    reference: left half is a normal `MeshRenderer` drawing world-space
    normals, right half is the ray traced result colored by world-space
    geometric normals. The tori rotate so the two views can be seen staying
    in sync. Matching silhouettes and colors confirm correct intersections.
    The texture is also read back with `ReadPixels` and saved to
    `Output/rt-result.png`, verifying that natively written contents are
    visible to Unity.
- `Assets/Resources/Torus.obj`, `Sphere.obj` — Generated test meshes (torus:
  major radius 1.0, minor radius 0.4, 4096 triangles; UV sphere: radius 0.5,
  960 triangles).

## Results

Verified on Unity 6000.3.19f1 / macOS / Apple M4 Max:

- `SystemInfo.supportsRayTracing == false` on Metal (Unity API path is
  unavailable, as expected).
- Native `MTLDevice.supportsRaytracing == true`.
- Acceleration structure build from non-readable meshes: **OK** — the GPU
  buffers of a `Mesh` with `isReadable == false` can be consumed directly by
  `MTLAccelerationStructureTriangleGeometryDescriptor`.
- Instance acceleration structure (TLAS): **OK** — three instances over two
  BLASes, rebuilt every frame from scene transforms (animated instances stay
  in sync with the rasterized reference), with bindless per-instance buffer
  access for hit shading.
- Probe ray tests: **5/5 passed**, hit distances matching analytic values
  within 0.02 and hit instance indices matching the expected instances.
- Visual test: the ray traced image matches the rasterized reference in
  silhouette and normal color distribution (see `Images/comparison.png`).
- Direct `RenderTexture` write: **OK** — the native plugin writes the trace
  result straight into a Unity `RenderTexture` every frame with no CPU
  readback, and Unity can display and `ReadPixels` it afterwards.
- Render thread integration: **OK** — the per-frame TLAS rebuild and trace
  are encoded into Unity's current Metal command buffer from a
  `IssuePluginEventAndData` render event, with no `waitUntilCompleted` on
  the frame path. Display, `ReadPixels`, and animated instances all stay
  correctly ordered with Unity's rendering.

## How to run

1. Build the plugin: `NativePlugin/build.sh` (requires Xcode Command Line
   Tools; outputs `Assets/Plugins/macOS/libMetalRTTest.dylib`).
2. Open the project with Unity 6000.3.19f1 and enter play mode. Results are
   logged to the console with a `[MetalRT]` prefix, and images are written to
   `Output/`.

Note: the macOS editor never unloads native plugins, so restart the editor
after rebuilding the plugin.

## Notes for the next stage (URP cooperation)

All the building blocks for URP integration are now verified: acceleration
structures from engine meshes, TLAS instancing from scene transforms, direct
`RenderTexture` output, and render-thread scheduling inside Unity's command
stream. The remaining work is URP-specific plumbing — driving the plugin
event from a `ScriptableRenderPass` and consuming URP's cameras and render
targets instead of the test harness.
