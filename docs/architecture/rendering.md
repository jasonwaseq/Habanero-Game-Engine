# Rendering Pipeline

## Backend Choice

Habanero targets Vulkan through `ash` to maximize low-level control and demonstrate explicit GPU resource orchestration.

## Stages

1. **Extraction**: scene/world data transformed into `DrawPacket` arrays.
2. **Visibility**: frustum test and coarse culling.
3. **Pass planning**: frame graph chooses pass order and attachments.
4. **Submission**: backend records commands and submits to graphics queue.

## Deferred + Forward Hybrid

- depth prepass
- gbuffer pass
- lighting pass
- forward transparent pass
- post-process chain

## CPU/GPU Synchronization

- Use frame-in-flight resource rings to reduce stalls.
- Reuse command buffers and descriptor pools.
- Prefer buffered updates over immediate map/unmap churn.

## GPU Memory Management Strategy

- Suballocate large device-local heaps for static resources.
- Keep staging resources host-visible and recycled.
- Track usage/lifetimes per pass for transient render targets.
