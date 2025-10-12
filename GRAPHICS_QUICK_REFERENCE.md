# Graphics Module - Quick Reference

## For Integrating Graphics into Your Application

### Step 1: Setup Window and Surface
```rust
use winit::{event_loop::EventLoop, window::WindowBuilder};
use vulkano::{instance::Instance, swapchain::Surface};

let event_loop = EventLoop::new()?;
let window = WindowBuilder::new().build(&event_loop)?;
let surface = Surface::from_window(instance.clone(), window.clone())?;
```

### Step 2: Load Shader Module
```rust
use rust_gpu_chimera_demo::OTHER_SHADERS_SPIRV;
use vulkano::shader::ShaderModule;

let shader_module = unsafe {
    ShaderModule::from_bytes(device.clone(), OTHER_SHADERS_SPIRV)?
};
```

### Step 3: Create Graphics Renderer
```rust
use rust_gpu_chimera_demo::graphics::GraphicsRenderer;

let renderer = GraphicsRenderer::new(
    instance.clone(),
    surface.clone(),
    shader_module.clone(),  // vertex shader
    shader_module.clone(),  // fragment shader
)?;
```

### Step 4: Run Compute and Get Buffer
```rust
let (buffer_x, num_particles) = runner.run_compute_and_get_buffer(
    &mut a, &b, &c, &d,  // u32 buffers
    &mut x, &v,          // Vec2 buffers
)?;
```

### Step 5: Set Position Buffer
```rust
renderer.set_position_buffer(buffer_x, num_particles)?;
```

### Step 6: Render in Event Loop
```rust
event_loop.run(move |event, _, control_flow| {
    match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => {
                *control_flow = ControlFlow::Exit;
            }
            WindowEvent::RedrawRequested => {
                if let Err(e) = renderer.render_frame() {
                    eprintln!("Render error: {}", e);
                }
                window.request_redraw();
            }
            WindowEvent::Resized(size) => {
                renderer.set_window_resized();
                if let Err(e) = renderer.handle_resize([size.width, size.height]) {
                    eprintln!("Resize error: {}", e);
                }
            }
            _ => {}
        },
        _ => {}
    }
});
```

## API Reference

### `GraphicsRenderer`

#### Constructor
```rust
pub fn new(
    instance: Arc<Instance>,
    surface: Arc<Surface>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
) -> CrateResult<Self>
```

#### Methods
```rust
// Set which buffer to render
pub fn set_position_buffer(
    &mut self, 
    position_buffer: Subbuffer<[Vec2]>, 
    num_particles: usize
) -> CrateResult<()>

// Render one frame
pub fn render_frame(&mut self) -> CrateResult<()>

// Mark window as resized
pub fn set_window_resized(&mut self)

// Handle resize if needed
pub fn handle_resize(&mut self, new_dimensions: [u32; 2]) -> CrateResult<()>
```

### `VulkanoRunner` (New Methods)

```rust
// Run compute and return the x buffer for graphics
pub fn run_compute_and_get_buffer(
    &self,
    a: &mut [u32], b: &[u32], c: &[u32], d: &[u32],
    x: &mut [Vec2], v: &[Vec2],
) -> CrateResult<(Subbuffer<[Vec2]>, usize)>

// Get the device (for creating graphics resources)
pub fn device(&self) -> &Arc<Device>

// Get the memory allocator
pub fn memory_allocator(&self) -> &Arc<StandardMemoryAllocator>
```

## Coordinate System

Particle positions in the buffer should be in **normalized [0, 1] coordinates**:
- `(0.0, 0.0)` = top-left of screen
- `(1.0, 1.0)` = bottom-right of screen  
- `(0.5, 0.5)` = center of screen

The vertex shader automatically converts these to Vulkan clip space [-1, 1].

## Troubleshooting

### "No suitable device available for graphics"
- Ensure your system has a Vulkan-capable GPU
- Check that the surface/window was created correctly
- Verify swapchain extension (khr_swapchain) is supported

### "Vertex shader entry point 'main_vs' not found"
- Ensure you're using the correct SPIR-V module
- Check that shaders were compiled: `cargo build` should show "SPIRV entry points: ..., main_vs, main_fs, ..."

### Blank screen / no particles visible
- Verify `set_position_buffer()` was called with valid data
- Check that positions are in [0, 1] range
- Ensure `num_particles` > 0
- Call `window.request_redraw()` to trigger rendering

### Window resize causes crash
- Make sure to call `handle_resize()` on `WindowEvent::Resized`
- The swapchain must be recreated when window size changes

## Performance Tips

1. **Don't wait every frame**: Current implementation calls `wait(None)` which blocks. For better performance, implement proper fence tracking.

2. **Batch updates**: If particle positions change frequently, consider updating the buffer less often or use double buffering.

3. **Point size**: If you need larger particles, consider using a geometry shader to expand points to quads, or use instanced rendering.

4. **Resize handling**: Minimize resizes or debounce resize events to avoid constant swapchain recreation.

## Example: Animating Particles

```rust
// Run compute multiple times before rendering
for _ in 0..10 {
    let (buffer_x, num_particles) = runner.run_compute_and_get_buffer(
        &mut a, &b, &c, &d, &mut x, &v
    )?;
    
    renderer.set_position_buffer(buffer_x, num_particles)?;
    renderer.render_frame()?;
    
    std::thread::sleep(Duration::from_millis(16)); // ~60 FPS
}
```

## See Also

- `src/graphics/README.md` - Detailed architecture and design documentation
- `GRAPHICS_IMPLEMENTATION.md` - Complete implementation summary
- `shaders/src/lib.rs` - Shader source code with `main_vs` and `main_fs`
