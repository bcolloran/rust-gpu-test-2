# 🗡🩸🗡🩸🗡 Sharp bits 🗡🩸🗡🩸🗡
- had trouble using rust-gpu with glam's `Vec2` type in shared code -- boiled down to missing "bytemuck" feature in glam and spirv-std dependencies in shared and shaders crates. Remember to use that!
- also, best to use the same root workspace deps e.g. glam, spirv-std, bytemuck in all crates to avoid feature and version mismatches.



# Project Structure
```
rust-gpu-chimera-demo/
├── shaders/          # GPU shaders (compute + graphics)
│   └── src/
│       ├── lib.rs         # Main shader entry points
│       └── bindless.rs    # Bindless resource shaders
├── shared/           # Code and types shared between CPU and GPU
│   └── src/
│       ├── lib.rs
│       └── grid.rs        # GridCell and GridPushConstants types
├── src/
│   ├── graphics/     # Graphics rendering module
│   │   ├── device.rs      # Device/queue selection
│   │   ├── pipeline.rs    # Pipeline creation (grid + particles)
│   │   ├── renderer.rs    # Main renderer with dual pipelines
│   │   └── README.md      # Graphics module documentation
│   ├── runners/      # Compute pipeline runners
│   │   ├── vulkano.rs
│   │   └── vulkano_compute_chain.rs
│   ├── lib.rs
│   └── main.rs       # Demo application with windowing
└── build.rs          # Shader compilation orchestration
```

## `shared` Crate
It's important that types that are shared between the CPU and GPU have identical memory layouts. This crate is included as a dependency in both the `kernel` and main application crates.

Types in this crate will include things like: push constant structs, items used in data buffers, and any other data structures that need to be shared between host and device.

