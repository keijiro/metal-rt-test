# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release: hardware-accelerated path tracer for URP on macOS
  (Apple Silicon) built on Metal ray tracing, with URP Lit material
  support, conditional Shader Graph support (generated material
  evaluation compute shaders), punctual lights, alpha-tested shadows,
  an a-trous denoiser, automatic scene registration, and edit mode
  support.
