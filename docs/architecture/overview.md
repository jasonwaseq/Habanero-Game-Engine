# Engine Architecture Overview

## Subsystem Boundaries

- `engine-core`: lifecycle, plugin loading, schedule orchestration.
- `engine-ecs`: data-oriented runtime state.
- `engine-scene`: hierarchy and persistence format.
- `engine-render`: render extraction, frame graph planning, backend execution.
- `engine-assets`: import, cache, dependency metadata, hot reload source watches.
- `engine-editor`: tooling UI model and runtime bridge.

## Data Ownership

- Long-lived owners: `Engine`, `AssetManager`, renderer resource managers.
- Transient per-frame data: extracted draw packets, culling results, command lists.
- Shared access is explicit: immutable snapshots and channels over broad mutable sharing.

## Frame Lifecycle

1. Poll platform/input.
2. Run ECS/simulation schedule.
3. Produce render extraction packets from scene/world.
4. Culling and batching.
5. Render graph pass execution.
6. Audio/script/editor updates.
7. Present and collect frame stats.

## Plugin Model

Plugins implement `EnginePlugin` and receive mutable access during registration to:

- add systems/resources
- register asset importers
- attach editor tooling
