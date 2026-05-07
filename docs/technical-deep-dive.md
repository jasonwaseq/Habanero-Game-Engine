# Technical Deep Dive

## Ownership and Scheduling

`Engine` is the top-level owner of long-lived systems. Simulation and rendering are staged to reduce lock contention:

- simulation writes ECS state
- render extraction reads ECS and builds immutable packets
- renderer consumes packets and records GPU commands

This avoids broad `Arc<Mutex<T>>` sharing and keeps data movement explicit.

## Multithreading Strategy

- `JobSystem` wraps a rayon pool for work-stealing.
- Systems with no write conflicts can run in parallel scopes.
- Render backend submission remains ordered; extraction and culling are parallel-friendly.

## Asset Pipeline

- UUID-backed `AssetId` handles
- metadata table for dependency/source tracking
- cache of loaded bytes
- file watchers for hot-reload triggers

## Graphics Expansion Plan

- wire real swapchain image acquisition/present path
- implement descriptor allocator and pipeline cache
- add shadow map atlas and clustered/light lists
- improve render graph with explicit transient resource aliases
