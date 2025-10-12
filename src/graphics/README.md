# Graphics Module

This module provides a simple graphics pipeline for rendering Vec2 particle positions to the screen using Vulkan and rust-gpu.

## Overview

The graphics system is designed to be relatively independent from the compute pipeline while sharing the same Vulkan device and memory. It demonstrates how to:

1. **Create a window and swapchain** for presenting rendered images
2. **Use rust-gpu compiled shaders** (from `shaders/src/lib.rs`) in a graphics pipeline
3. **Read particle positions from a GPU buffer** that was populated by compute shaders
4. **Render points directly on screen** without CPU-side vertex buffers

## Architecture

The module is organized into several sub-modules:

- **`device.rs`**: Device and queue selection for graphics operations
- **`pipeline.rs`**: Graphics pipeline creation and descriptor set management
- **`renderer.rs`**: Main rendering loop, swapchain management, and command buffer creation

## How It Works

### 1. Shader Pipeline

The shaders in `shaders/src/lib.rs` contain:

- **`main_vs` (Vertex Shader)**: Reads Vec2 positions from a storage buffer (binding 0, set 0), converts them from [0, 1] normalized coordinates to Vulkan clip space [-1, 1], and outputs colored vertices
- **`main_fs` (Fragment Shader)**: Simply outputs the interpolated color from the vertex shader

### 2. Data Flow

```
Compute Shaders → "x" Buffer (Vec2 positions) → Vertex Shader → Screen
```

1. Compute shaders (in `VulkanoRunner`) populate the "x" buffer with particle positions
2. The graphics pipeline binds this buffer as a storage buffer (descriptor set 0, binding 0)
3. The vertex shader reads one position per invocation using `vertex_index`
4. Each position is rendered as a point on screen

### 3. Key Concepts

#### Storage Buffer in Vertex Shader
Unlike traditional graphics pipelines that use vertex buffers, we use a **storage buffer** (SSBO) accessed in the vertex shader. This allows:
- Direct GPU-to-GPU data flow (no CPU readback needed)
- Flexibility in how we interpret the data
- The same buffer to be used by both compute and graphics

#### Normalized Coordinates
Particle positions are stored in [0, 1] range where:
- `(0, 0)` = top-left corner
- `(1, 1)` = bottom-right corner

The vertex shader converts these to Vulkan's clip space [-1, 1].

#### Point Topology
The pipeline uses `PrimitiveTopology::PointList`, meaning each vertex index generates one point on screen. The size of the point is typically 1 pixel.

### 4. Rendering Process

```rust
// In your application:
let (buffer_x, num_particles) = runner.run_compute_and_get_buffer(...)?;
renderer.set_position_buffer(buffer_x, num_particles)?;
renderer.render_frame()?;
```

Each frame:
1. **Acquire** a swapchain image
2. **Execute** the command buffer that:
   - Begins render pass (clears to dark blue)
   - Binds graphics pipeline
   - Binds descriptor set (with position buffer)
   - Draws N points (where N = number of particles)
   - Ends render pass
3. **Present** the image to the screen

## Integration Example

Here's how to integrate graphics rendering with the existing compute pipeline:

```rust
use rust_gpu_chimera_demo::{VulkanoRunner, graphics::GraphicsRenderer};
use vulkano::{instance::Instance, swapchain::Surface};

// 1. Create compute runner
let runner = VulkanoRunner::new(shader_buffers)?;

// 2. Create graphics renderer (requires window/surface)
let event_loop = EventLoop::new()?;
let window = WindowBuilder::new().build(&event_loop)?;
let surface = Surface::from_window(instance.clone(), window.clone())?;

let shader_module = /* load shader module from SPIR-V */;
let vs = shader_module.clone();
let fs = shader_module.clone();

let mut renderer = GraphicsRenderer::new(instance, surface, vs, fs)?;

// 3. Run compute shaders and get the position buffer
let (buffer_x, num_particles) = runner.run_compute_and_get_buffer(
    &mut a, &b, &c, &d, &mut x, &v
)?;

// 4. Set the buffer in the renderer
renderer.set_position_buffer(buffer_x, num_particles)?;

// 5. Render frames in event loop
event_loop.run(move |event, _, control_flow| {
    match event {
        Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
            renderer.render_frame().unwrap();
        }
        Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
            renderer.set_window_resized();
            renderer.handle_resize([size.width, size.height]).unwrap();
        }
        _ => {}
    }
});
```

## Shader Details

### Vertex Shader (`main_vs`)

```rust
#[spirv(vertex)]
pub fn main_vs(
    #[spirv(vertex_index)] vert_idx: i32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] positions: &[Vec2],
    #[spirv(position)] builtin_pos: &mut Vec4,
    out_color: &mut Vec4,
) {
    let idx = vert_idx as usize;
    let pos = positions[idx];
    
    // Convert [0, 1] to [-1, 1] clip space
    let clip_pos = pos * 2.0 - Vec2::ONE;
    *builtin_pos = clip_pos.extend(0.0).extend(1.0);
    
    // Color based on particle index
    let color_factor = (idx % 256) as f32 / 256.0;
    *out_color = Vec4::new(
        0.3 + 0.7 * color_factor,
        0.5 + 0.5 * (1.0 - color_factor),
        0.8,
        1.0,
    );
}
```

### Fragment Shader (`main_fs`)

```rust
#[spirv(fragment)]
pub fn main_fs(in_color: Vec4, output: &mut Vec4) {
    *output = in_color;
}
```

## Performance Considerations

- **Simple synchronization**: Current implementation waits for each frame to complete before starting the next. For better performance, you'd want to implement proper fence management per swapchain image.
  
- **Point size**: Each particle is rendered as a 1-pixel point. For larger/variable sizes, you'd need to use a geometry shader or instanced quads.

- **No vertex buffers**: Using storage buffers instead of vertex buffers has some overhead but provides more flexibility.

## Future Enhancements

Possible improvements:
1. **Better synchronization**: Track fences per swapchain image for proper frame pacing
2. **Variable point sizes**: Use geometry shader to expand points to quads
3. **More visual effects**: Add trails, colors based on velocity, etc.
4. **Multiple render passes**: Implement post-processing effects
5. **Push constants**: Pass viewport size, time, etc. to shaders

## Learning Resources

- [Vulkano Tutorial](https://vulkano.rs/) - The renderer structure is inspired by vulkano's windowing chapter
- [Vulkan Guide](https://vkguide.dev/) - General Vulkan concepts
- [rust-gpu](https://github.com/EmbarkStudios/rust-gpu) - Writing shaders in Rust
