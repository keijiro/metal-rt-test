# metal-rt-test

A Mac hardware ray tracing (MetalRT) path tracer for Unity, aiming at an
HDRP-path-tracing-like feature for URP.

![comparison](Images/comparison.png)

*Left: URP real-time rasterization. Right: the same view path traced by a
URP camera whose renderer has `MetalRTPathTracerFeature`. The checkerboard
floor uses a Shader Graph: URP rasterizes it with the graph's own shader
while the path tracer evaluates a compute shader automatically generated
from the same graph — both views match. The path traced view adds soft
shadows, GI color bleeding, emissive lighting, and mirror reflections.*

## Overview

This project verifies, step by step, that hardware ray tracing on macOS
(Apple Silicon) can cooperate with Unity — starting from building
acceleration structures out of non-readable Unity meshes, through a
progressive path tracer that consumes real URP/Lit materials inside Unity's
own Metal command stream, **conditional Shader Graph support** (material
evaluation as Unity-compiled compute shaders, generatable from Shader Graph
assets), and now **full URP integration**: the path tracer runs as a
`ScriptableRendererFeature` — add the feature to a URP renderer and any
camera using it gets path traced output, recorded through a RenderGraph
unsafe pass and composited onto the camera color target.

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
  `MTLDevice` and command queue through `IUnityGraphicsMetalV2` and
  implements the ray tracing side as runtime-compiled MSL compute kernels:
  - **Acceleration structures**: per-mesh BLASes are built directly from the
    GPU buffers of non-readable Unity meshes
    (`Mesh.GetNativeVertexBufferPtr` / `GetNativeIndexBufferPtr` — no CPU
    copy of the geometry ever exists), then combined into a TLAS with
    per-instance transforms.
  - **Wavefront pipeline in phases**: the per-frame pipeline is split into
    plugin render events — Begin (TLAS rebuild + `RayGen`), then per bounce
    `Intersect` + `GeomPrep` (attribute interpolation + default URP Lit
    surface evaluation) and `Shade` (NEE + BSDF sampling), then `Resolve`
    (progressive accumulation + ACES tonemap into a Unity `RenderTexture`).
    Hit records, hit attributes, and surface records live in Unity-created
    `GraphicsBuffer`s shared with the plugin by native pointer, so **Unity
    compute shaders dispatched between Intersect and Shade can evaluate
    materials**, overwriting surface records for their material indices.
  - **URP/Lit BSDF**: Lambert diffuse + GGX specular with the URP
    metallic/smoothness convention, cosine / GGX-NDF importance sampling
    with lobe selection, and next event estimation over the scene's
    punctual lights — directional, point, and spot with URP's attenuation
    conventions (inverse square with smooth range window, squared
    smoothstep spot falloff), intensity premultiplied by pi. Shadow rays
    evaluate the base map alpha of alpha-clipped Lit materials (up to four
    transparent layers), so cutout materials cast correct shadows.
  - **Denoiser**: during early accumulation (< 64 frames) the resolve
    stage runs an edge-avoiding a-trous wavelet filter over the
    albedo-demodulated irradiance, guided by primary-hit albedo and normal,
    then remodulates — texture detail stays sharp while GI noise smooths
    out. Converged frames and the analytic test modes bypass the filter.
  - **Bindless resources** (Metal 3): mesh buffers are referenced by
    `gpuAddress` and material base maps by `gpuResourceID` from
    plain-buffer-resident tables, with explicit `useResource` residency.
  - **Render thread integration**: each phase commits Unity's current
    command buffer (`CommitCurrentCommandBuffer`) and encodes into its own
    command buffer on Unity's queue, so native kernels and Unity compute
    dispatches stay GPU-ordered on one command stream with no CPU blocking.
    (Creating encoders on Unity's own command buffer is not safe here:
    Unity's compute encoders stay open across plugin events.)
- `Assets/Editor/ShaderGraphComputeGen.cs` — The Shader Graph to compute
  shader generator. It obtains the generated shader text through
  `ShaderGraphImporter.GetShaderText` (internal API via reflection), slices
  out the `SurfaceDescriptionFunction`, its graph functions, the
  `UnityPerMaterial` cbuffer, and texture declarations, then wraps them in a
  compute kernel that maps path tracer hit attributes (uv0/uv1, vertex
  color, world-space geometry, view direction, time) to
  `SurfaceDescriptionInputs`. `Alpha` / `AlphaClipThreshold` outputs map to
  the surface record for alpha clipping. Texture sampling macros are
  redefined to their LOD variants for compute. Graphs requiring unsupported
  inputs (screen position, scene color/depth, etc.) are rejected — this is
  the "conditional" support boundary. An `AssetPostprocessor` regenerates
  the compute shader whenever a graph under `Assets/Shaders` is reimported.
- `Assets/Scripts/MetalRTPathTracer.cs` +
  `Assets/Scripts/MetalRTPathTracerFeature.cs` — The URP integration. The
  runtime core owns the progressive result texture, the shared wavefront
  buffers, and the material evaluation compute list, and records the phase
  pipeline into a command buffer. The renderer feature drives it from a
  RenderGraph **unsafe pass** (the same escape hatch DLSS/FSR2-style native
  integrations use): `CommandBufferHelpers.GetNativeCommandBuffer` provides
  the raw command buffer for `IssuePluginEventAndData` and the material
  dispatches, then the result is blitted onto the camera color target.
  Accumulation restarts automatically when the camera moves.
- `Assets/Scripts/MetalRTSceneRegistry.cs` — Automatic scene registration:
  scans the scene's MeshRenderers and registers BLASes for unique meshes,
  the material table, per-frame instance descriptors, and generated Shader
  Graph computes (resolved by naming convention, shader "X/Name" ->
  Resources "NameGen") with a generic property/keyword binder.
- `Assets/Scripts/PathTracerTest.cs` — The test harness. It builds a static
  URP scene at runtime (registration is automatic); the left camera
  rasterizes it normally while the right camera uses the renderer with the
  path tracer feature. The floor uses `Assets/Shaders/TestGraph.shadergraph`
  rasterized by URP on the left and evaluated by its generated compute
  shader in the path tracer on the right. Hand-written
  SurfaceDescription-style computes drive additional test materials
  (procedural pattern, vertex color with a keyword variant, alpha cutout).
- `Assets/Scripts/MetalRTPlugin.cs` — P/Invoke interop and the event data
  blob writer (a small ring of unmanaged blobs passes per-frame camera,
  lighting, and instance transforms to the render thread).

## Verification

Analytic tests (logged as PASS/FAIL to the console on play):

- **Probe rays**: 5 world-space rays against the TLAS with analytically
  known hit distances and instance indices.
- **T1 direct lighting**: with environment off and a single bounce, a floor
  pixel must equal `albedo/pi * lightColor * cos(theta)` where the albedo is
  the replicated checker base map sample. Measured relative error:
  **0.03 %** (native URP Lit evaluation path).
- **T2 furnace test**: a convex Lambertian sphere (albedo 0.5) in a uniform
  environment must return exactly `rho * E` (zero-variance for a convex
  body). Measured relative error: **0.00 %**.
- **T3 Unity-compiled material evaluation**: the hand-written
  SurfaceDescription-style compute shader overrides the floor material and
  must reproduce the C#-replicated procedural pattern. Measured relative
  error: **0.00 %** (validates the wavefront interop end to end).
- **T4 Shader Graph generated material**: the compute shader generated from
  `TestGraph.shadergraph` evaluates the floor and must match the
  C#-replicated base map sample times tint. The tint is changed for the
  test only — the generated compute reads material properties live while
  the native fallback keeps its setup snapshot, so the test passes only
  when the generated kernel really runs. Measured relative error:
  **0.48 %**.

- **T5 vertex color input**: a runtime-built quad (made non-readable on
  upload) with UNorm8 vertex colors, evaluated by a compute shader reading
  `VertexColor`; the pixel at a triangle centroid must equal the
  barycentric average of its corner colors. Measured relative error:
  **0.72 %**.
- **T6 alpha clipping**: a cutout quad clips half its checker cells via
  `Alpha` / `AlphaClipThreshold`; clipped cells must show the environment
  behind (**0.00 %**) and solid cells behave like a flat furnace surface
  (**0.00 %**).
- **T7 keyword variants**: toggling a keyword on the material switches the
  evaluated compute variant (**0.42 %**).
- **T8 punctual lights**: point and spot lighting on a flat receiver match
  URP's attenuation formulas analytically (**0.01 %** each).
- **T9 alpha-tested shadows**: swapping `_Cutoff` around a half-alpha
  occluder toggles a point light's shadow between fully transparent
  (**0.04 %**) and fully opaque (**exact zero**). Clipping that exists only
  in compute-evaluated materials still shadows as opaque (limitation).

All of the above run through the URP renderer feature (RenderGraph unsafe
pass), not a standalone dispatch path.

Visual verification: matching composition and shadow directions against the
URP raster reference, the Shader Graph floor matching between the raster
(graph shader) and path traced (generated compute) views, plus
path-tracing-only effects (emissive light bleed, GI color bleeding,
physically correct mirror reflections).

## How to run

Requirements: an Apple Silicon Mac with hardware ray tracing (M3 or later)
and Unity 6000.3.19f1. The prebuilt native plugin is included; to rebuild
it, run `NativePlugin/build.sh` (Xcode Command Line Tools) and restart the
editor (the macOS editor never unloads native plugins).

**Sample scene**: open `Assets/Scenes/Sample.unity` and press play. The
left half shows URP rasterization, the right half the path traced view of
the same scene (`MetalRTPathTracerRunner` registers the scene's renderers
and lights automatically; tweak bounces/exposure on the runner object).
The scene is fully dynamic: moving, adding, removing, or re-materialing
objects and editing lights restarts the accumulation automatically.

**Edit mode**: the runner is `[ExecuteAlways]`, so the path traced view
also works without entering play mode. Since the editor only redraws on
demand, accumulation normally progresses while you interact with the
editor; enable **Continuous Edit Refresh** on the runner to keep the
player loop ticking so the image converges on its own.

**Test suite**: enter play mode in a scene *without* a
`MetalRTPathTracerRunner` (e.g., an empty scene) — the test harness
bootstraps itself, runs the probe and T1-T9 analytic checks (logged with a
`[MetalRT]` prefix), and then renders progressively, saving a converged
frame to `Output/rt-result.png`.

## Possible future work

The original roadmap (keyword variants, automatic scene registration,
punctual lights, alpha-tested shadows, denoising) is complete. Natural next
directions:

- Transparency and refraction (transmission BSDF, non-opaque intersection)
- Emissive mesh light sampling (NEE with MIS instead of BSDF-only)
- Skinned/dynamic meshes (BLAS refit per frame)
- Multiple UV channels beyond uv1, HDR environment maps
- A higher-quality denoiser (temporal reprojection, or ML-based)
