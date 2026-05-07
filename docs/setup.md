# Setup

## Prerequisites

- Rust stable toolchain (`rustup default stable`)
- Vulkan SDK + validation layers installed
- Graphics driver supporting Vulkan 1.2+

## Commands

```bash
cargo check
cargo test
cargo run -p runtime-demo
cargo run -p editor-app
```

## Recommended Dev Flags

- `RUST_LOG=info` for runtime tracing
- `RUST_BACKTRACE=1` for debugging panics

## Project Conventions

- subsystem crates under `crates/`
- applications under `apps/`
- architecture docs under `docs/architecture/`
