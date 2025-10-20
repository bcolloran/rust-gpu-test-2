# 🗡🩸🗡🩸🗡 Sharp bits 🗡🩸🗡🩸🗡
- had trouble using rust-gpu with glam's `Vec2` type in shared code -- boiled down to missing "bytemuck" feature in glam and spirv-std dependencies in shared and shaders crates. Remember to use that!
- also, best to use the same root workspace deps e.g. glam, spirv-std, bytemuck in all crates to avoid feature and version mismatches.



# Project Structure

```
rust-gpu-chimera-demo/
├── kernel/           # Compute kernel logic and entrypoints
│   └── src/
│       └── lib.rs
├── shared/           # Code that runs on both the CPU and GPU
│   └── src/
│       └── lib.rs
├── src/
│   ├── runners/      # Code that runs on the CPU/host and interfaces with the GPU
│   │   ├── ash.rs    # DEPRECATED, reference only
│   │   ├── vulkano.rs
│   │   └── vulkano-bindless.rs
│   ├── lib.rs
│   └── main.rs       # Demo application binary
└── build.rs          # Kernel compilation orchestration
```

