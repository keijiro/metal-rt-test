using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using static MetalRTTest.MetalRTPlugin;

namespace MetalRTTest {

// Scans the scene's MeshRenderers and registers everything the path tracer
// needs: BLASes for each unique (non-readable) mesh, the material table,
// per-frame TLAS instance descriptors, and — for materials whose shader is
// not URP/Lit — the compute shader generated from their Shader Graph
// (resolved by naming convention: shader "X/Name" -> Resources "NameGen").
public sealed class MetalRTSceneRegistry
{
    // Public interface

    public bool Build(MetalRTPathTracer tracer)
    {
        var renderers = UnityEngine.Object.FindObjectsByType<MeshRenderer>
          (FindObjectsSortMode.InstanceID);

        foreach (var renderer in renderers)
        {
            var filter = renderer.GetComponent<MeshFilter>();
            var mesh = filter != null ? filter.sharedMesh : null;
            var material = renderer.sharedMaterial;
            if (mesh == null || material == null) continue;

            var meshIndex = RegisterMesh(mesh);
            if (meshIndex < 0) continue;

            var materialIndex = RegisterMaterial(material, tracer);
            _instances.Add((meshIndex, materialIndex, renderer.transform));
        }

        if (_instances.Count == 0)
        {
            Debug.LogError("[MetalRT] No renderers registered");
            return false;
        }
        if (_instances.Count > MaxInstances)
        {
            Debug.LogError($"[MetalRT] Too many instances " +
                           $"({_instances.Count} > {MaxInstances})");
            return false;
        }

        if (!UploadMaterials()) return false;

        Debug.Log($"[MetalRT] Scene registry: {_instances.Count} instances, " +
                  $"{_meshes.Count} meshes, {_materials.Count} materials " +
                  "(auto-registered from MeshRenderers)");
        return true;
    }

    public int MaterialIndexOf(Material material)
      => _materialIndices.TryGetValue(material, out var i) ? i : -1;

    public int InstanceIndexOf(Transform transform)
      => _instances.FindIndex(e => e.transform == transform);

    public MetalRTPathTracer.MaterialCompute ComputeOf(Material material)
      => _materialComputes.TryGetValue(material, out var c) ? c : null;

    // Re-uploads the native material table (call after changing Lit
    // material properties consumed by the native evaluation).
    public bool RefreshMaterials() => UploadMaterials();

    // Per-frame TLAS instance descriptors from the current transforms.
    public InstanceDesc[] MakeDescs()
    {
        var descs = new InstanceDesc[_instances.Count];
        for (var i = 0; i < _instances.Count; i++)
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

    // Private members

    readonly Dictionary<Mesh, int> _meshIndices = new();
    readonly List<Mesh> _meshes = new();
    readonly Dictionary<Material, int> _materialIndices = new();
    readonly List<Material> _materials = new();
    readonly Dictionary<Material, MetalRTPathTracer.MaterialCompute>
      _materialComputes = new();
    readonly List<(int mesh, int material, Transform transform)>
      _instances = new();

    // Mesh registration (BLAS construction from GPU buffers)

    int RegisterMesh(Mesh mesh)
    {
        if (_meshIndices.TryGetValue(mesh, out var cached)) return cached;

        var posStream = mesh.GetVertexAttributeStream(VertexAttribute.Position);
        var posOffset = mesh.GetVertexAttributeOffset(VertexAttribute.Position);
        var posFormat = mesh.GetVertexAttributeFormat(VertexAttribute.Position);
        var posDim = mesh.GetVertexAttributeDimension(VertexAttribute.Position);
        var stride = mesh.GetVertexBufferStride(posStream);

        if (posFormat != VertexAttributeFormat.Float32 || posDim != 3 ||
            mesh.GetTopology(0) != MeshTopology.Triangles ||
            mesh.GetBaseVertex(0) != 0)
        {
            Debug.LogWarning($"[MetalRT] Unsupported mesh layout: {mesh.name}");
            _meshIndices.Add(mesh, -1);
            return -1;
        }

        var normalOffset = AttributeOffset(mesh, VertexAttribute.Normal,
                                           posStream, 3);
        var tangentOffset = AttributeOffset(mesh, VertexAttribute.Tangent,
                                            posStream, 4);
        var uvOffset = AttributeOffset(mesh, VertexAttribute.TexCoord0,
                                       posStream, 2);
        var uv1Offset = AttributeOffset(mesh, VertexAttribute.TexCoord1,
                                        posStream, 2);
        var (colorOffset, colorFormat) = ColorAttribute(mesh, posStream);

        var indexCount = mesh.GetIndexCount(0);
        var indexStart = mesh.GetIndexStart(0);
        var is16Bit = mesh.indexFormat == IndexFormat.UInt16;
        var indexSize = is16Bit ? 2u : 4u;

        var ret = MetalRT_AddMesh
          (mesh.GetNativeVertexBufferPtr(posStream), (uint)stride,
           (uint)posOffset, normalOffset, tangentOffset, uvOffset,
           uv1Offset, colorOffset, colorFormat,
           mesh.GetNativeIndexBufferPtr(),
           is16Bit ? 0u : 1u, indexStart * indexSize, indexCount / 3);

        if (ret < 0)
        {
            Debug.LogError($"[MetalRT] BLAS build error {ret} for " +
                           $"{mesh.name}: {LastError}");
            _meshIndices.Add(mesh, -1);
            return -1;
        }

        Debug.Log($"[MetalRT] BLAS #{ret} built from non-readable mesh " +
                  $"{mesh.name} ({indexCount / 3} triangles, isReadable = " +
                  $"{mesh.isReadable}): OK");

        _meshIndices.Add(mesh, ret);
        _meshes.Add(mesh);
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

    static (uint offset, uint format) ColorAttribute(Mesh mesh, int stream)
    {
        if (!mesh.HasVertexAttribute(VertexAttribute.Color) ||
            mesh.GetVertexAttributeStream(VertexAttribute.Color) != stream ||
            mesh.GetVertexAttributeDimension(VertexAttribute.Color) != 4)
            return (NoAttribute, 0);
        var offset = (uint)mesh.GetVertexAttributeOffset(VertexAttribute.Color);
        return mesh.GetVertexAttributeFormat(VertexAttribute.Color) switch
        {
            VertexAttributeFormat.Float32 => (offset, 0u),
            VertexAttributeFormat.UNorm8 => (offset, 1u),
            _ => (NoAttribute, 0u)
        };
    }

    // Material registration

    int RegisterMaterial(Material material, MetalRTPathTracer tracer)
    {
        if (_materialIndices.TryGetValue(material, out var cached))
            return cached;

        var index = _materials.Count;
        _materials.Add(material);
        _materialIndices.Add(material, index);

        // Non-Lit shaders: attach the compute shader generated from the
        // material's Shader Graph when one exists.
        var shaderName = material.shader.name;
        if (shaderName != "Universal Render Pipeline/Lit")
        {
            var lastSegment = shaderName.Substring
              (shaderName.LastIndexOf('/') + 1);
            var cs = Resources.Load<ComputeShader>(lastSegment + "Gen");
            if (cs != null)
            {
                var compute = MakeShaderGraphCompute(cs, material, index);
                _materialComputes.Add(material, compute);
                tracer.MaterialComputes.Add(compute);
                Debug.Log($"[MetalRT] Shader Graph compute attached: " +
                          $"{shaderName} -> {lastSegment}Gen " +
                          $"(material {index})");
            }
            else
            {
                Debug.LogWarning($"[MetalRT] No generated compute for " +
                                 $"shader {shaderName}; material {index} " +
                                 "falls back to native Lit evaluation");
            }
        }
        return index;
    }

    // Uploads the native material table (URP Lit property scrape).
    bool UploadMaterials()
    {
        var descs = new MaterialDesc[_materials.Count];
        for (var i = 0; i < _materials.Count; i++)
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
                cutoff = mat.IsKeywordEnabled("_ALPHATEST_ON") &&
                         mat.HasProperty("_Cutoff") ?
                  mat.GetFloat("_Cutoff") : -1,
                hasBaseMap = tex != null ? 1u : 0u,
                texture = tex != null ?
                  tex.GetNativeTexturePtr().ToInt64() : 0
            };
        }

        var ret = MetalRT_SetMaterials(descs, descs.Length);
        if (ret != 0)
        {
            Debug.LogError($"[MetalRT] Material table error {ret}: " +
                           LastError);
            return false;
        }
        return true;
    }

    // Generic property/keyword binder for generated Shader Graph computes.

    static MetalRTPathTracer.MaterialCompute MakeShaderGraphCompute
      (ComputeShader cs, Material mat, int materialIndex)
    {
        var kernel = cs.FindKernel("EvaluateMaterial");
        var shader = mat.shader;
        var floats = new List<string>();
        var colors = new List<(string, bool)>();
        var vectors = new List<string>();
        var textures = new List<(string, string)>();

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
                BindMaterialKeywords(cb, cs, mat);
            }
        };
    }

    // Synchronizes the compute shader's local keywords with the material's
    // enabled keywords (Shader Graph keyword variants). Uses the raw
    // keyword string list so keywords not declared by the material's own
    // shader are honored too.
    public static void BindMaterialKeywords(CommandBuffer cb,
                                            ComputeShader cs, Material mat)
    {
        var enabled = mat.shaderKeywords;
        foreach (var kw in cs.keywordSpace.keywords)
            cb.SetKeyword(cs, kw, Array.IndexOf(enabled, kw.name) >= 0);
    }

    static Texture DefaultTexture(string name) => name switch
    {
        "black" => Texture2D.blackTexture,
        "bump" => Texture2D.normalTexture,
        "gray" or "grey" => Texture2D.grayTexture,
        _ => Texture2D.whiteTexture
    };
}

} // namespace MetalRTTest
