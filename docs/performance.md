# Performance and Profiling

## Primary Bottlenecks to Track

- draw call count
- CPU simulation cost
- render extraction/culling cost
- GPU frame time and stalls
- allocation churn per frame

## Strategies Implemented

- contiguous component storage in ECS
- staged frame pipeline with explicit extraction
- render packet batching hook points
- rayon-backed job execution for independent tasks

## Benchmark Plan

- ECS iteration throughput (`benches/ecs_stress.rs`)
- transform propagation scale test
- culling throughput on synthetic scenes
- frame prep breakdown (extract/cull/submit)

## Tooling

- `tracing` spans for frame subsystems
- debug metrics in runtime/editor (`fps`, `entity_count`, `draw_calls`)
- validation builds with Vulkan debug/validation layers enabled
