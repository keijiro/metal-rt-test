#!/bin/sh
# Builds the Metal RT test plugin into Assets/Plugins/macOS.
# Note: The Unity editor does not unload native plugins; restart the editor
# after rebuilding.
set -e

cd "$(dirname "$0")"

UNITY_APP=/Applications/Unity/Hub/Editor/6000.3.19f1/Unity.app
PLUGIN_API="$UNITY_APP/Contents/PluginAPI"
OUT_DIR=../Assets/Plugins/macOS

mkdir -p "$OUT_DIR"

xcrun clang++ -std=c++17 -fobjc-arc -O2 -shared \
  -arch arm64 -mmacosx-version-min=13.0 \
  -isystem "$PLUGIN_API" \
  -framework Metal -framework Foundation \
  -o "$OUT_DIR/libMetalRTTest.dylib" MetalRTPlugin.mm

echo "Built $OUT_DIR/libMetalRTTest.dylib"
