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

Current foundation:

- Camera abstraction (orthographic + perspective)
- Scene extraction into draw packets
- Render graph scaffold with passes:
  - depth prepass
  - gbuffer
  - lighting
  - transparent forward
  - post process

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
cargo test
cargo run -p runtime-demo
cargo run -p editor-app
```

## Demo Targets

- 2D sprite batching stress scene
- 3D lit scene baseline
- ECS throughput stress test
- Asset hot reload loop for shaders/textures
- Runtime HUD metrics (FPS/entity/draw-call/memory placeholders)

## Resume-Worthy Scope

Habanero demonstrates practical systems engineering across:

- low-level graphics architecture (Vulkan-oriented renderer design)
- data-oriented ECS and scheduling
- multithreaded frame preparation
- asset pipeline + hot reload
- tooling/editor runtime integration
- benchmarking, profiling, and technical documentation
