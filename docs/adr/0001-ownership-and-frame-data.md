# ADR 0001: Ownership and Frame Data Flow

## Status

Accepted

## Context

Game engines frequently share mutable state across rendering, gameplay, physics, and tools. In Rust, this can create coarse locks if ownership is not designed carefully.

## Decision

- Keep runtime-wide owners centralized in `Engine`.
- Move per-frame render data into extracted packet buffers.
- Use message passing and staged boundaries between systems.

## Consequences

- Reduced lock contention risk.
- Clearer boundaries between simulation and rendering.
- Slight overhead of extraction copies, offset by better cache behavior and parallelism.
