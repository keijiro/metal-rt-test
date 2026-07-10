using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using UnityEditor;
using UnityEngine;

namespace MetalRTTest.Editor {

// Generates a material evaluation compute shader for the Metal RT path
// tracer from a Shader Graph asset. The graph's SurfaceDescriptionFunction
// (and its supporting code) is extracted from the generated shader text and
// wrapped in a compute kernel that maps path tracer hit attributes to
// SurfaceDescriptionInputs. Graphs whose inputs go beyond the supported set
// (UVs, world-space geometry, view direction, time) are rejected — this is
// the "conditional Shader Graph support" boundary.
public static class ShaderGraphComputeGen
{
    const string GraphPath = "Assets/Shaders/TestGraph.shadergraph";
    const string OutputPath = "Assets/Resources/TestGraphGen.compute";

    [MenuItem("MetalRT/Generate Compute From Test Graph")]
    public static void GenerateTestGraph()
    {
        var error = Generate(GraphPath, OutputPath);
        if (error == null)
            Debug.Log($"[MetalRT] Generated {OutputPath} from {GraphPath}");
        else
            Debug.LogError($"[MetalRT] Generation failed: {error}");
    }

    // Supported SurfaceDescriptionInputs fields and the expressions that
    // fill them from the hit attributes (see the kernel template below for
    // the local variable names).
    static readonly Dictionary<string, string> InputMap = new()
    {
        { "uv0", "float4(attr.uvView.xy, 0, 0)" },
        { "uv1", "float4(attr.uvView.zw, 0, 0)" },
        { "VertexColor", "attr.color" },
        { "TangentSpaceNormal", "float3(0, 0, 1)" },
        { "WorldSpaceNormal", "nrm" },
        { "WorldSpaceTangent", "tang" },
        { "WorldSpaceBiTangent", "bitan" },
        { "WorldSpacePosition", "attr.position.xyz" },
        { "AbsoluteWorldSpacePosition", "attr.position.xyz" },
        { "WorldSpaceViewDirection", "normalize(attr.viewDir.xyz)" },
        { "ObjectSpaceNormal", null },   // unsupported
        { "ScreenPosition", null },      // unsupported
        { "TimeParameters", "_TimeParams.xyz" },
        { "FaceSign", "1" },
    };

    public static string Generate(string graphPath, string outputPath)
    {
        string text;
        try
        {
            text = GetShaderText(graphPath);
        }
        catch (Exception e)
        {
            return $"GetShaderText failed: {e.Message}";
        }
        if (string.IsNullOrEmpty(text)) return "empty generated shader text";

        // Locate the first pass that has a surface description function
        // (Universal Forward) and slice out the pieces we need.
        var fn = text.IndexOf
          ("SurfaceDescription SurfaceDescriptionFunction(",
           StringComparison.Ordinal);
        if (fn < 0) return "SurfaceDescriptionFunction not found";

        var cbStart = text.LastIndexOf("CBUFFER_START(UnityPerMaterial)", fn,
                                       StringComparison.Ordinal);
        if (cbStart < 0) return "UnityPerMaterial cbuffer not found";
        var cbEnd = text.IndexOf("CBUFFER_END", cbStart, StringComparison.Ordinal)
                    + "CBUFFER_END".Length;
        var cbuffer = text.Substring(cbStart, cbEnd - cbStart);
        cbuffer = string.Join("\n", cbuffer.Split('\n')
          .Where(l => !l.Contains("UNITY_TEXTURE_STREAMING_DEBUG_VARS")));

        var giMark = text.IndexOf("// Graph Includes", cbEnd,
                                  StringComparison.Ordinal);
        if (giMark < 0) return "Graph Includes marker not found";
        var texDecls = string.Join("\n",
          text.Substring(cbEnd, giMark - cbEnd).Split('\n')
              .Select(l => l.Trim())
              .Where(l => Regex.IsMatch
                 (l, @"^(TEXTURE2D|TEXTURE3D|TEXTURECUBE|TEXTURE2D_ARRAY|SAMPLER)\(")));

        var gfMark = text.IndexOf("// Graph Functions", giMark,
                                  StringComparison.Ordinal);
        if (gfMark < 0) return "Graph Functions marker not found";
        var gfEnd = text.IndexOf("// Custom interpolators pre vertex", gfMark,
                                 StringComparison.Ordinal);
        if (gfEnd < 0)
            gfEnd = text.IndexOf("// Graph Vertex", gfMark,
                                 StringComparison.Ordinal);
        if (gfEnd < 0) return "end of Graph Functions not found";
        var functions = text.Substring(gfMark, gfEnd - gfMark);

        var inStructIdx = text.LastIndexOf("struct SurfaceDescriptionInputs",
                                           fn, StringComparison.Ordinal);
        var inputsStruct = ExtractBlock(text, inStructIdx);

        var surfStructIdx = FindSurfaceStruct(text, fn);
        if (surfStructIdx < 0) return "SurfaceDescription struct not found";
        var surfStruct = ExtractBlock(text, surfStructIdx);

        var function = ExtractBlock(text, fn);

        // Build the input adapter and validate the graph's requirements.
        var inputFields = ParseStructFields(inputsStruct);
        var assigns = new StringBuilder();
        foreach (var (type, name) in inputFields)
        {
            if (!InputMap.TryGetValue(name, out var expr) || expr == null)
                return $"unsupported SurfaceDescriptionInputs field: " +
                       $"{type} {name}";
            assigns.AppendLine($"    IN.{name} = {expr};");
        }

        // Output mapping from the SurfaceDescription fields.
        var surfFields = ParseStructFields(surfStruct)
                         .Select(f => f.name).ToHashSet();
        if (!surfFields.Contains("BaseColor"))
            return "SurfaceDescription has no BaseColor output";

        var normalCode = "    float3 nWS = nrm;";
        if (surfFields.Contains("NormalWS"))
            normalCode = "    float3 nWS = normalize(d.NormalWS);";
        else if (surfFields.Contains("NormalTS"))
            normalCode = "    float3 nWS = normalize(tang * d.NormalTS.x + " +
                         "bitan * d.NormalTS.y + nrm * d.NormalTS.z);";

        var emission = surfFields.Contains("Emission") ? "d.Emission" : "0";
        var metallic = surfFields.Contains("Metallic") ? "d.Metallic" : "0";
        var smoothness = surfFields.Contains("Smoothness") ? "d.Smoothness"
                                                           : "0.5";
        var alpha = surfFields.Contains("Alpha") ? "d.Alpha" : "1";
        var threshold = surfFields.Contains("AlphaClipThreshold") ?
          "d.AlphaClipThreshold" : "-1";

        var code = $@"// Auto-generated by ShaderGraphComputeGen from {graphPath}. Do not edit.
#pragma kernel EvaluateMaterial

#include ""Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl""
#include ""Packages/com.unity.render-pipelines.core/ShaderLibrary/Texture.hlsl""
#include ""Packages/com.unity.shadergraph/ShaderGraphLibrary/Functions.hlsl""

// Compute shaders have no derivatives; force LOD 0 sampling for the graph
// code, which accesses textures through these macros.
#undef SAMPLE_TEXTURE2D
#define SAMPLE_TEXTURE2D(textureName, samplerName, coord2) SAMPLE_TEXTURE2D_LOD(textureName, samplerName, coord2, 0)
#undef SAMPLE_TEXTURECUBE
#define SAMPLE_TEXTURECUBE(textureName, samplerName, coord3) SAMPLE_TEXTURECUBE_LOD(textureName, samplerName, coord3, 0)

// ---- Extracted from the Shader Graph generated shader ----

{cbuffer}

{texDecls}

{inputsStruct}

{functions}

{surfStruct}

{function}

// ---- Path tracer adapter ----

struct HitAttributes
{{
    float4 position;
    float4 normal;
    float4 tangent;
    float4 uvView;
    float4 viewDir;
    float4 color;
    uint4 meta;
}};

struct SurfaceRecord
{{
    float4 baseColor;
    float4 normal;
    float4 emission;
    float4 params;
}};

StructuredBuffer<HitAttributes> _Attributes;
RWStructuredBuffer<SurfaceRecord> _Surfaces;
uint _Width, _Height, _MaterialIndex;
float4 _TimeParams;

[numthreads(8, 8, 1)]
void EvaluateMaterial(uint3 id : SV_DispatchThreadID)
{{
    if (id.x >= _Width || id.y >= _Height) return;
    uint idx = id.y * _Width + id.x;

    HitAttributes attr = _Attributes[idx];
    if (attr.meta.y == 0 || attr.meta.x != _MaterialIndex) return;

    float3 nrm = normalize(attr.normal.xyz);
    float3 tang = attr.tangent.xyz;
    float3 bitan = normalize(cross(nrm, tang)) * attr.tangent.w;

    SurfaceDescriptionInputs IN = (SurfaceDescriptionInputs)0;
{assigns.ToString().TrimEnd()}

    SurfaceDescription d = SurfaceDescriptionFunction(IN);

{normalCode}

    SurfaceRecord s;
    s.baseColor = float4(d.BaseColor, 1);
    s.normal = float4(nWS, 0);
    s.emission = float4({emission}, 0);
    s.params = float4({metallic}, {smoothness}, {alpha}, {threshold});
    _Surfaces[idx] = s;
}}
";
        System.IO.File.WriteAllText(outputPath, code);
        AssetDatabase.ImportAsset(outputPath);
        return null;
    }

    // Finds "struct SurfaceDescription" (not ...Inputs) before the function.
    static int FindSurfaceStruct(string text, int before)
    {
        var idx = before;
        while (true)
        {
            idx = text.LastIndexOf("struct SurfaceDescription", idx - 1,
                                   StringComparison.Ordinal);
            if (idx < 0) return -1;
            var next = text[idx + "struct SurfaceDescription".Length];
            if (next == '\n' || next == '\r' || next == ' ' || next == '{')
                return idx;
        }
    }

    // Extracts a brace-balanced block starting at the given declaration
    // index (includes a trailing semicolon for struct declarations).
    static string ExtractBlock(string text, int start)
    {
        var open = text.IndexOf('{', start);
        var depth = 0;
        for (var i = open; i < text.Length; i++)
        {
            if (text[i] == '{') depth++;
            else if (text[i] == '}')
            {
                depth--;
                if (depth == 0)
                {
                    var end = i + 1;
                    if (end < text.Length && text[end] == ';') end++;
                    return text.Substring(start, end - start);
                }
            }
        }
        return null;
    }

    static List<(string type, string name)> ParseStructFields(string block)
    {
        var fields = new List<(string, string)>();
        var body = block.Substring(block.IndexOf('{') + 1);
        foreach (Match m in Regex.Matches(body, @"^\s*(\w+)\s+(\w+)\s*;",
                                          RegexOptions.Multiline))
            fields.Add((m.Groups[1].Value, m.Groups[2].Value));
        return fields;
    }

    // Regenerates the compute shader whenever a shader graph under
    // Assets/Shaders is (re)imported.
    sealed class Postprocessor : AssetPostprocessor
    {
        static void OnPostprocessAllAssets(string[] imported, string[] deleted,
                                           string[] moved, string[] movedFrom)
        {
            foreach (var path in imported)
            {
                if (!path.StartsWith("Assets/Shaders/") ||
                    !path.EndsWith(".shadergraph")) continue;
                var captured = path;
                // Defer: importing the generated asset from within a
                // postprocessor callback is not allowed.
                EditorApplication.delayCall += () =>
                {
                    var name = System.IO.Path.GetFileNameWithoutExtension
                      (captured);
                    var output = $"Assets/Resources/{name}Gen.compute";
                    var error = Generate(captured, output);
                    if (error == null)
                        Debug.Log($"[MetalRT] Regenerated {output} " +
                                  $"from {captured}");
                    else
                        Debug.LogError($"[MetalRT] Compute generation failed " +
                                       $"for {captured}: {error}");
                };
            }
        }
    }

    // Calls the internal ShaderGraphImporter.GetShaderText via reflection.
    static string GetShaderText(string path)
    {
        var importer = AppDomain.CurrentDomain.GetAssemblies()
          .Select(a => a.GetType("UnityEditor.ShaderGraph.ShaderGraphImporter"))
          .FirstOrDefault(t => t != null);
        if (importer == null)
            throw new InvalidOperationException("ShaderGraphImporter not found");

        const BindingFlags flags = BindingFlags.Static | BindingFlags.Public |
                                   BindingFlags.NonPublic;
        var method = importer.GetMethods(flags).FirstOrDefault
          (m => m.Name == "GetShaderText" && m.GetParameters().Length == 4 &&
                m.GetParameters()[3].IsOut);
        if (method == null)
            throw new InvalidOperationException("GetShaderText overload not found");

        var args = new object[] { path, null, null, null };
        return (string)method.Invoke(null, args);
    }
}

} // namespace MetalRTTest.Editor
