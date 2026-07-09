Shader "MetalRT/WorldNormal"
{

HLSLINCLUDE

#include "UnityCG.cginc"

void Vert(float4 position : POSITION,
          float3 normal : NORMAL,
          out float4 outPosition : SV_Position,
          out float3 outNormal : TEXCOORD0)
{
    outPosition = UnityObjectToClipPos(position);
    outNormal = UnityObjectToWorldNormal(normal);
}

float4 Frag(float4 position : SV_Position,
            float3 normal : TEXCOORD0) : SV_Target
{
    return float4(normalize(normal) * 0.5 + 0.5, 1);
}

ENDHLSL

    SubShader
    {
        Pass
        {
            Name "WorldNormal"
            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            ENDHLSL
        }
    }
}
