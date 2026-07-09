# metal-rt-test

A Mac hardware ray tracing (MetalRT) path tracer for Unity, aiming at an
HDRP-path-tracing-like feature for URP.

![comparison](Images/comparison.png)

*Left: URP real-time rasterization. Right: the Metal RT path tracer rendering
the same scene with the same URP/Lit materials (soft shadows, GI color
bleeding, emissive lighting, and mirror reflections).*

## Overview

This project verifies, step by step, that hardware ray tracing on macOS
(Apple Silicon) can cooperate with Unity — starting from building
acceleration structures out of non-readable Unity meshes and ending (so far)
at a progressive path tracer that consumes real URP/Lit materials and runs
inside Unity's own Metal command stream.

Current stage: **stage 1 — Lit-materials-only path tracer**. The next stage
is conditional Shader Graph support through wavefront-style material
evaluation with Unity-compiled compute shaders.

## Background

As of Unity 6.3 (6000.3), Unity's native ray tracing APIs
(`RayTracingAccelerationStructure`, `RayTracingShader`, inline `RayQuery`) are
not supported on Metal — `SystemInfo.supportsRayTracing` returns `false`, and
Metal support is a long-term item with no announced release. The
`UnifiedRayTracing` API falls back to a compute shader implementation on
Metal, which is not hardware ray tracing.

Therefore, the only way to use hardware ray tracing on Mac is a native plugin
that calls the Metal ray tracing APIs (`MTLAccelerationStructure` +
`metal::raytracing::intersector`) directly. That plugin is the core of this
project.

## How it works

- `NativePlugin/MetalRTPlugin.mm` — The native plugin. It obtains Unity's
  `MTLDevice` through `IUnityGraphicsMetalV1` and implements the whole path
  tracer as runtime-compiled MSL compute kernels:
  - **Acceleration structures**: per-mesh BLASes are built directly from the
    GPU buffers of non-readable Unity meshes
    (`Mesh.GetNativeVertexBufferPtr` / `GetNativeIndexBufferPtr` — no CPU
    copy of the geometry ever exists), then combined into a TLAS with
    per-instance transforms.
  - **Wavefront-style pipeline**: per frame, `RayGen` emits jittered camera
    rays, then `Intersect` / `Shade` alternate per bounce over path state
    buffers, and `Resolve` accumulates into a progressive HDR buffer and
    writes the tonemapped (ACES) linear result into a Unity `RenderTexture`.
    Surface evaluation is isolated behind a `SurfaceData` boundary in the
    Shade kernel — the exact seam where stage 2 will substitute
    Unity-compiled Shader Graph material evaluation.
  - **URP/Lit BSDF**: Lambert diffuse + GGX specular with the URP
    metallic/smoothness convention (`roughness = (1 - smoothness)^2`,
    `F0 = lerp(0.04, baseColor, metallic)`), cosine / GGX-NDF importance
    sampling with lobe selection, and next event estimation for the
    directional light via shadow rays. Directional light intensity follows
    Unity's punctual light convention (premultiplied by pi).
  - **Bindless resources** (Metal 3): mesh buffers are referenced by
    `gpuAddress` and material base maps by `gpuResourceID` from
    plain-buffer-resident tables, with explicit `useResource` residency.
  - **Render thread integration**: the per-frame TLAS rebuild and the full
    kernel pipeline are encoded into Unity's current Metal command buffer
    from a `CommandBuffer.IssuePluginEventAndData` render event
    (`EndCurrentCommandEncoder` + `CurrentCommandBuffer`), so everything is
    GPU-ordered with Unity's rendering and the CPU never blocks.
- `Assets/Scripts/PathTracerTest.cs` — The test harness. It builds a static
  URP scene at runtime (floor, checker-textured torus, mirror-metal sphere,
  white torus, emissive sphere — all real "Universal Render Pipeline/Lit"
  materials), registers meshes/materials/instances with the plugin, and
  shows the URP raster view (left) and the path traced view (right) side by
  side. The path traced result is displayed through URP itself (a second
  camera rendering a fullscreen quad) so both halves share the same color
  pipeline.
- `Assets/Scripts/MetalRTPlugin.cs` — P/Invoke interop and the event data
  blob writer (a small ring of unmanaged blobs passes per-frame camera,
  lighting, and instance transforms to the render thread).

## Verification

Analytic tests (logged as PASS/FAIL to the console on play):

- **Probe rays**: 5 world-space rays against the TLAS with analytically
  known hit distances and instance indices.
- **T1 direct lighting**: with environment off and a single bounce, a floor
  pixel must equal `albedo/pi * lightColor * cos(theta)`. Measured relative
  error: **0.00 %**.
- **T2 furnace test**: a convex Lambertian sphere (albedo 0.5) in a uniform
  environment must return exactly `rho * E` (zero-variance for a convex
  body). Measured relative error: **0.00 %**.

Visual verification: matching composition and shadow directions against the
URP raster reference, plus path-tracing-only effects (emissive light bleed,
GI color bleeding, physically correct mirror reflections), with matching
overall brightness (floor pixel values agree within ~1 %).

## How to run

1. Build the plugin: `NativePlugin/build.sh` (requires Xcode Command Line
   Tools; outputs `Assets/Plugins/macOS/libMetalRTTest.dylib`).
2. Open the project with Unity 6000.3.19f1 and enter play mode. Test results
   are logged to the console with a `[MetalRT]` prefix; a converged frame is
   saved to `Output/rt-result.png`.

Note: the macOS editor never unloads native plugins, so restart the editor
after rebuilding the plugin.

## Roadmap

- **Stage 2 — conditional Shader Graph support**: replace the `SurfaceData`
  evaluation with per-material compute shaders generated from Shader Graph's
  `SurfaceDescription` functions and compiled by Unity for Metal
  (wavefront-style deferred material evaluation).
- **URP plumbing**: drive the render event from a `ScriptableRenderPass`
  (RenderGraph) and consume URP cameras and render targets directly.
