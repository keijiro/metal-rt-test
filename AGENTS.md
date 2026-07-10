# Workflow Instructions for Agents

## Project Structure

The Packages directory contains the primary UPM package developed in this
project: `jp.keijiro.urp-metal-path-tracer`. The repository root is the
development and test Unity project for the package.

The repository root contains README.md and CHANGELOG.md. Equivalent files
exist inside the package, so keep them synchronized with the root copies
whenever you update the root-level documents. This project currently has no
LICENSE file (intentionally deferred).

Package layout:

- `Packages/jp.keijiro.urp-metal-path-tracer/Runtime` — runtime scripts and
  the prebuilt native plugin (`Plugins/libMetalPathTracer.dylib`)
- `Packages/jp.keijiro.urp-metal-path-tracer/Editor` — the Shader Graph to
  compute shader generator
- `NativePlugin/` — native plugin source; `build.sh` rebuilds the dylib
  into the package

## Project-Specific Rules

- The macOS editor never unloads native plugins. After rebuilding the
  dylib, restart the Unity editor before testing.
- The analytic test suite (probe rays and T1-T9) runs by entering play
  mode in a scene *without* a MetalRTPathTracerRunner (e.g., an empty
  scene). `Assets/Scenes/Sample.unity` is the interactive demo instead.
- The shared GPU struct layouts (HitAttributes, SurfaceRecord, etc.) are
  duplicated across the native plugin, the generator template, and the
  hand-written test compute shaders. When changing them, update every copy
  and regenerate the generated compute assets (reimport the graph or use
  the context menu).

## Action Definition: Updating the Changelog

Updating the changelog means bringing the [Unreleased] section of
CHANGELOG.md up to date. Review git commits made since the previous release
and append the relevant changes to that section.

The [Unreleased] section may already include manually written text.
Proofread it and adjust the wording so that it matches the newly added
entries. Add the [Unreleased] section if it is missing.

## Action Definition: Preparing a Package Release

Preparing a package release means refreshing the UPM package data inside
the Packages directory so it is ready for a new version.

Perform the following tasks:

- Bump the version field in package.json.
- Update the `_upm` element in package.json as described below.
- Change the [Unreleased] heading in CHANGELOG.md to the new version and
  today's date.
- Synchronize the package copies of README.md and CHANGELOG.md.
- Commit the changes and create a git tag for the new version number.

## About the `_upm` Element in package.json

The `_upm` element in package.json contains only the `changelog` entry.
Copy the latest version section from CHANGELOG.md into that entry, but
remove the section heading (version number and a date) and convert the
content to Unity Rich Text. Use `<b>` tags for headings and `<br>` for line
breaks.
