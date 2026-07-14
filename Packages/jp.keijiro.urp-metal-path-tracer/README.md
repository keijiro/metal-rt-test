# URP Metal Path Tracer

**URP Metal Path Tracer** (`jp.keijiro.urp-metal-path-tracer`) is a
hardware-accelerated path tracer for the Universal Render Pipeline on macOS,
built directly on the Metal ray tracing APIs.

![comparison](Images/comparison.png)

*The sample scene (a Cornell-box arrangement with a smoothness ladder of
metallic and dielectric spheres). Left: URP real-time rasterization. Right:
the same view path traced by a URP camera with the path tracer renderer
feature — emissive panel lighting, red/green GI color bleeding, and soft
shadows. The checkerboard floor is a Shader Graph, rasterized with the
graph's own shader on the left and evaluated by an automatically generated
compute shader on the right.*

## Why this exists

As of Unity 6.3, Unity's native ray tracing APIs
(`RayTracingAccelerationStructure`, `RayTracingShader`, inline `RayQuery`)
are not supported on Metal. This package implements the whole path tracing
stack in a native plugin instead: acceleration structures are built directly
from Unity mesh GPU buffers (no CPU copies, works with non-readable meshes),
rays are traced with `metal::raytracing::intersector`, and everything is
scheduled inside Unity's own Metal command stream through a RenderGraph
unsafe pass — the same integration pattern DLSS-style native plugins use.

## Features

- **Progressive path tracing** of URP scenes: URP/Lit materials
  (base map, metallic/smoothness, emission, alpha clipping), punctual
  lights (directional/point/spot with URP's attenuation conventions),
  flat ambient environment, ACES tonemapping
- **Conditional Shader Graph support**: an editor tool extracts the
  `SurfaceDescriptionFunction` from a Shader Graph and generates a material
  evaluation compute shader that runs inside the wavefront loop. Graphs
  driven by UVs (uv0/uv1), vertex color, world-space geometry, view
  direction, time, and keyword variants are supported; screen-space inputs
  are rejected
- **Alpha-tested shadows**, **ML denoising** during early accumulation
  with [Intel Open Image Denoise] running on its Metal backend (the
  official OIDN binaries are downloaded into the project `Library`
  folder on first use, so the package itself stays small), **automatic
  scene registration** (MeshRenderer scan with add/remove/move detection
  and automatic accumulation restart), and **edit mode support**

[Intel Open Image Denoise]: https://www.openimagedenoise.org/

## System requirements

- macOS on Apple Silicon with hardware ray tracing (M3 or later)
- Unity 6000.3
- Universal Render Pipeline

## Installation

Add the package to your project via the Package Manager. For a Git-based
install, add this line to `Packages/manifest.json`:

```
"jp.keijiro.urp-metal-path-tracer": "https://github.com/keijiro/metal-rt-test.git?path=Packages/jp.keijiro.urp-metal-path-tracer"
```

## How to use

1. Create a Universal Renderer asset with **MetalRTPathTracerFeature**
   added, and register it in your URP asset's renderer list.
2. Assign that renderer to the camera that should show the path traced
   image (`UniversalAdditionalCameraData.SetRenderer`). Setting the
   camera's culling mask to Nothing skips the redundant raster pass.
3. Add a **MetalRTPathTracerRunner** component to any object in the scene.
   It registers the scene's MeshRenderers and lights automatically and
   keeps them in sync; bounces and exposure are exposed in the inspector.
4. For Shader Graph materials, select the `.shadergraph` asset and run
   **Assets > URP Metal Path Tracer > Generate Material Compute**. The
   generated compute shader (in `Assets/Resources`) is picked up by naming
   convention and regenerated automatically whenever the graph reimports.

The runner also works in edit mode (`[ExecuteAlways]`): accumulation
progresses on editor repaints, or continuously when **Continuous Edit
Refresh** is enabled.

This repository doubles as the development project: open
`Assets/Scenes/Sample.unity` for a ready-made split-view demo, or enter
play mode in an empty scene to run the analytic test suite (probe rays and
T1-T9 radiometric checks, logged with a `[MetalRT]` prefix).

## Limitations

- Opaque and alpha-clipped materials only (no transparency/refraction)
- Alpha clipping computed by Shader Graph materials shadows as opaque
  (native Lit alpha-clip materials cast correct cutout shadows)
- Up to 16 instances, 4 lights, uv0/uv1; submesh 0 only
- Emissive surfaces are sampled by BSDF rays only (no NEE/MIS for mesh
  lights); area lights are not supported

## How it works

- `NativePlugin/MetalRTPlugin.mm` — runtime-compiled MSL kernels: per-mesh
  BLASes from Unity mesh GPU buffers combined into a per-frame TLAS;
  a wavefront pipeline (RayGen, per-bounce Intersect + GeomPrep + Shade,
  Resolve) over buffers shared with Unity compute shaders; bindless
  mesh/texture access (Metal 3 `gpuAddress` / `gpuResourceID`); phase
  events commit Unity's current command buffer and encode into their own
  command buffers on Unity's queue (`IUnityGraphicsMetalV2`); the OIDN
  device is created on the same queue and its filter runs over shared
  Metal buffers (color plus albedo/normal guides), so denoising is
  sequenced by commit order with no extra synchronization
- `Runtime/MetalRTPathTracer.cs` — owns the GPU-facing state and records
  the phase pipeline; restarts accumulation when the camera, scene
  content, or settings change
- `Runtime/MetalRTSceneRegistry.cs` — scans MeshRenderers, builds the
  material table from URP/Lit properties, and attaches generated Shader
  Graph computes with a generic property/keyword binder
- `Runtime/MetalRTPathTracerFeature.cs` — the RenderGraph unsafe pass that
  drives the tracer and blits the result onto the camera color target
- `Editor/ShaderGraphComputeGen.cs` — the Shader Graph to compute shader
  generator (via `ShaderGraphImporter.GetShaderText`), including keyword
  variant emission and the input validation that defines the "conditional"
  support boundary

Verification is analytic: direct lighting, furnace tests, material
evaluation, keyword variants, punctual lights, and alpha-tested shadows are
all checked against closed-form expectations (errors of 1% or less; several
tests are exact by construction).
