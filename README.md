# Habanero Engine

Habanero Engine is a custom Rust 2D/3D game engine focused on modern architecture, ECS-driven gameplay, Vulkan rendering, and editor-first tooling. The project is built as a modular Cargo workspace to mirror real engine team boundaries and scale into advanced rendering and runtime systems.

## Why Rust for Engine Development

- Predictable ownership/lifetimes reduce memory leaks and use-after-free classes of bugs common in C++ engines.
- Zero-cost abstractions and explicit data layouts support performance-critical code.
- Strong type system improves subsystem API contracts (render graph resources, ECS queries, asset handles).
- Tooling (`cargo`, `clippy`, `rustfmt`, docs) improves maintainability for large systems projects.

### Tradeoffs vs C++

- **Pros**: safer concurrency, fewer UB footguns, stronger refactoring support.
- **Cons**: steeper lifetime ergonomics in some engine patterns, slower compile times, smaller ecosystem for specialized middleware.
- **Architectural impact**: favor immutable frame data handoff, command buffers, and message passing over broad shared mutable state.

## Workspace Layout

- `crates/engine-core`: loop, plugin interface, scheduler/job system
- `crates/engine-ecs`: sparse-set ECS with change detection and event bus
- `crates/engine-scene`: hierarchy + transform graph + scene serialization
- `crates/engine-render`: Vulkan-oriented renderer abstraction, frame graph, culling hooks
- `crates/engine-assets`: async-ready asset cache + metadata + hot-reload watcher
- `crates/engine-audio`: playback abstraction and positional audio base
- `crates/engine-physics`: integration layer (world step API boundary)
- `crates/engine-scripting`: Rhai scripting host and engine API bridge points
- `crates/engine-editor`: editor data model for hierarchy/inspector/stats panels
- `apps/runtime-demo`: runtime vertical-slice demo
- `apps/editor`: editor bootstrap demo
- `docs/`: architecture, rendering pipeline, ECS, performance, setup

## Architecture

Frame flow:

1. Platform input/events
2. ECS/system schedule
3. Render extraction
4. Visibility/culling
5. Render graph pass execution
6. Presentation + frame metrics

The engine keeps long-lived ownership in `engine-core`, while frame-level packets are transient and copied in cache-friendly contiguous arrays for renderer submission.

## Rendering Pipeline (current + target)

Current foundation (implemented and running on Vulkan):

- Camera abstraction (orthographic + perspective + orbiting look-at)
- Scene extraction into draw packets with per-instance color
- Frustum culling against the camera view-projection
- GPU-instanced draw of the extracted scene: per-instance model matrix + color
  uploaded to a persistently-mapped vertex buffer, view-projection and light
  direction supplied via push constants
- Directional lighting with depth test/write into the G-buffer + present target
- Dynamic viewport/scissor and full swapchain recreation on window resize
- Discrete/integrated GPU selection that is safe on hybrid-graphics laptops
- Render graph scaffold describing the deferred pass order:
  - depth prepass, gbuffer, lighting, shadow map, ssao, bloom,
    transparent forward, post process

Target expansion:

- Vulkan swapchain + command allocator pools
- deferred shading with G-buffer lifetime tracking
- shadow atlas and cascaded directional lights
- PBR material pipeline
- HDR + bloom + SSAO post chain
- GPU instancing and multi-draw submission

## ECS Design

- Entity IDs as compact `u64`
- Sparse-set component stores by component type
- Query APIs for sequential/parallel iteration
- Change detection for incremental systems
- Parent-child references in transforms

This balances cache locality and runtime flexibility while keeping archetype migration complexity low in the first milestone.

## Performance Strategy

- Minimize sync points using staged frame boundaries
- Prepare renderer packets in parallel before backend submission
- Use culling to reduce draw calls
- Keep components contiguous for vectorized iteration potential
- Track frame time, draw calls, and entity counts in editor/runtime stats

See `docs/performance.md` for profiling and benchmark guidance.

## Build & Run

```bash
cargo check
cargo test                                   # 24 tests across all crates

# Interactive windowed demo (Vulkan on by default).
# Recommended in release for a smooth, high-FPS flythrough:
cargo run --release -p runtime-demo
cargo run --release -p runtime-demo -- --stress=50000

# Headless / reproducible benchmark (no window, writes a perf report CSV):
cargo run --release -p runtime-demo -- --no-vulkan --frames=600 --stress=20000

cargo run -p editor-app
```

Runtime demo flags:

- `--stress=<N>`  number of animated entities (default 12000, capped at 200000)
- `--frames=<N>`  render N frames, print a p50/p95/p99 report, write
  `assets/processed/perf_report.csv`, then exit (great for CI/benchmarks)
- `--no-vulkan`   force the CPU/headless path (no GPU presentation)

Optional Vulkan debug env vars (debug builds): `HBN_ENABLE_VALIDATION_CALLBACK=1`
routes validation messages to the tracing log, `HBN_ENABLE_DEBUG_LABELS=1` adds
GPU command labels for RenderDoc/Nsight captures.

## Demo

The runtime demo renders an animated "galaxy": thousands of GPU-instanced, lit
cubes arranged in concentric rings plus a central rising helix. Every entity is
driven by the ECS each frame — `Spin` rotates orientation, `Orbit` revolves the
ring about the world axis — while an automatic camera orbits the scene. The title
bar reports live telemetry:

```
Habanero [Vulkan] | 84 FPS | 11.9 ms (p50 11.9 / p99 18.7) | entities 20000 | visible 15792 | culled 21.0%
```

This single scene exercises ECS scale, multithreaded simulation, frustum
culling, GPU instancing, directional lighting, and live profiling at once.

## Demo Targets

- Animated instanced 3D scene (`runtime-demo`) with live perf metrics
- ECS throughput stress test (`cargo run -p runtime-demo --bin ecs_stress`)
- 3D scene baseline (`cargo run -p runtime-demo --bin scene3d_demo`)
- 2D sprite batch (`cargo run -p runtime-demo --bin sprite_demo`)
- Asset hot reload loop (`cargo run -p runtime-demo --bin hot_reload_demo`)

## Next Implementation Steps

Recently completed:

- **Real scene draw path** — GPU-instanced draws are issued from the extracted,
  culled render packets (per-instance model + color), replacing the fullscreen
  debug pass as the primary path (the fullscreen pass remains as a safe fallback).
- **Performance instrumentation** — p50/p95/p99 frame-time percentiles plus
  reproducible CSV capture (`--frames` benchmark mode).
- **Robust demo scenario** — automated camera flythrough, configurable entity
  presets, swapchain-resize resilience, and a reproducible performance report.

Still ahead:

- **Deferred lighting completion**: split G-buffer and lighting into separate subpasses/pipelines, sample G-buffer in lighting pass, and add multi-light support.
- **GPU resource system**: add per-frame descriptor set ring, transient attachment allocator, and pipeline cache serialization.
- **Render graph execution**: move pass order/resources from hardcoded sequence to explicit graph nodes and dependencies.
- **Scalable ECS queries**: add multi-component query joins and system dependency graph with conflict detection.

## Resume-Worthy Scope

Habanero demonstrates practical systems engineering across:

- low-level graphics architecture (Vulkan-oriented renderer design)
- data-oriented ECS and scheduling
- multithreaded frame preparation
- asset pipeline + hot reload
- tooling/editor runtime integration
- benchmarking, profiling, and technical documentation
