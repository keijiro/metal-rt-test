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
  `MTLDevice` through `IUnityGraphicsMetalV1`, builds a primitive acceleration
  structure directly from the mesh GPU buffers, and dispatches compute kernels
  (compiled at runtime from embedded MSL source) that trace rays with
  `raytracing::intersector`. All GPU work runs synchronously on a private
  command queue, independent of Unity's render thread.
- `Assets/Scripts/MetalRayTracingTest.cs` — The test driver. It requires no
  scene setup (it bootstraps itself via `RuntimeInitializeOnLoadMethod`) and:
  - Loads the torus mesh, which is non-readable by default as an imported
    model asset, and passes its native vertex/index buffer pointers
    (`Mesh.GetNativeVertexBufferPtr` / `GetNativeIndexBufferPtr`) to the
    plugin — no CPU-side mesh data is ever touched.
  - Runs probe rays with analytically derived expectations for the torus
    (hit distances, misses through the hole) and logs PASS/FAIL.
  - Traces a full frame from the scene camera and shows it next to a
    rasterized reference: left half is a normal `MeshRenderer` drawing
    object-space normals, right half is the ray traced result colored by
    geometric normals. Matching silhouettes and colors confirm correct
    intersections. The image is also saved to `Output/rt-result.png`.
- `Assets/Resources/Torus.obj` — Generated test mesh (major radius 1.0, minor
  radius 0.4, 4096 triangles).

## Results

Verified on Unity 6000.3.19f1 / macOS / Apple M4 Max:

- `SystemInfo.supportsRayTracing == false` on Metal (Unity API path is
  unavailable, as expected).
- Native `MTLDevice.supportsRaytracing == true`.
- Acceleration structure build from a non-readable mesh: **OK** — the GPU
  buffers of a `Mesh` with `isReadable == false` can be consumed directly by
  `MTLAccelerationStructureTriangleGeometryDescriptor`.
- Probe ray tests: **5/5 passed**, hit distances matching analytic values
  within 0.02.
- Visual test: the ray traced image matches the rasterized reference in
  silhouette and normal color distribution (see `Images/comparison.png`).

## How to run

1. Build the plugin: `NativePlugin/build.sh` (requires Xcode Command Line
   Tools; outputs `Assets/Plugins/macOS/libMetalRTTest.dylib`).
2. Open the project with Unity 6000.3.19f1 and enter play mode. Results are
   logged to the console with a `[MetalRT]` prefix, and images are written to
   `Output/`.

Note: the macOS editor never unloads native plugins, so restart the editor
after rebuilding the plugin.

## Notes for the next stage (URP cooperation)

- Write results directly into a `RenderTexture` native pointer instead of
  reading back to the CPU.
- Use an instance acceleration structure (TLAS) for multiple meshes and
  world-space rays.
- Integrate with the render thread via `CommandBuffer.IssuePluginEventAndData`.
