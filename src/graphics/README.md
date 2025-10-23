# Graphics Module

This module provides graphics pipelines for rendering both a grid heatmap and Vec2 particle positions to the screen using Vulkan and rust-gpu.

## Overview

The graphics system is designed to be relatively independent from the compute pipeline while sharing the same Vulkan device and memory. It demonstrates how to:

1. **Create a window and swapchain** for presenting rendered images
2. **Use rust-gpu compiled shaders** (from `shaders/src/lib.rs`) in graphics pipelines
3. **Read particle positions and grid data from GPU buffers** that were populated by compute shaders
4. **Render a grid heatmap** as a background using instanced quads
5. **Render particle points on top** without CPU-side vertex buffers

## Architecture

The module is organized into several sub-modules:

- **`device.rs`**: Device and queue selection for graphics operations
- **`pipeline.rs`**: Graphics pipeline creation and descriptor set management
- **`renderer.rs`**: Main rendering loop, swapchain management, and command buffer creation

## How It Works

### 1. Dual Pipeline Architecture

The renderer uses **two separate graphics pipelines** that render in sequence:

1. **Grid Pipeline**: Renders a heatmap background using instanced quads
2. **Particle Pipeline**: Renders individual particles as points on top

### 2. Shader Entry Points

The shaders in `shaders/src/lib.rs` contain:

#### Grid Rendering Shaders
- **`grid_vs` (Vertex Shader)**: Reads GridCell data from storage buffer (binding 1, set 0), generates instanced quads (6 vertices per grid cell), and passes mass values to the fragment shader
- **`grid_fs` (Fragment Shader)**: Converts mass values [0, 1] to grayscale colors

#### Particle Rendering Shaders
- **`main_vs` (Vertex Shader)**: Reads Vec2 positions from storage buffer (binding 0, set 0), converts them from [0, 1] normalized coordinates to Vulkan clip space [-1, 1]
- **`main_fs` (Fragment Shader)**: Outputs a constant white color for particles

### 3. Data Flow

```
Compute Shaders → Grid Buffer (GridCell data) → Grid Pipeline → Background Heatmap
                ↓
                "x" Buffer (Vec2 positions) → Particle Pipeline → Foreground Points
```

1. Compute shaders populate the "grid" buffer with GridCell data (mass, velocity)
2. Compute shaders populate the "x" buffer with particle positions
3. Grid pipeline renders the heatmap first (background)
4. Particle pipeline renders points on top (foreground)

### 4. Key Concepts

#### Storage Buffers in Shaders
Unlike traditional graphics pipelines that use vertex buffers, we use **storage buffers** (SSBOs) accessed in vertex shaders. This allows:
- Direct GPU-to-GPU data flow (no CPU readback needed)
- Flexibility in how we interpret the data
- The same buffers to be used by both compute and graphics

#### Instanced Quad Rendering for Grid
The grid pipeline uses **instanced drawing** with procedurally generated geometry:
- Each grid cell is one instance
- The vertex shader generates 6 vertices per instance (2 triangles = 1 quad)
- Vertex positions are calculated based on `vertex_index` and `instance_index`
- **Push constants** pass grid dimensions to the shader
- Mass values are read from the GridCell buffer and used for coloring

#### Normalized Coordinates
Both particles and grid cells use [0, 1] normalized coordinates where:
- `(0, 0)` = top-left corner
- `(1, 1)` = bottom-right corner

The vertex shaders convert these to Vulkan's clip space [-1, 1].

#### Rendering Topologies
- **Grid Pipeline**: Uses `PrimitiveTopology::TriangleList` for instanced quads
- **Particle Pipeline**: Uses `PrimitiveTopology::PointList` for individual pixel points

### 5. Rendering Process

Each frame:
1. **Acquire** a swapchain image
2. **Execute** the command buffer that:
   - Begins render pass (clears to dark blue)
   - **Renders grid heatmap**:
     - Binds grid pipeline
     - Binds descriptor set (with GridCell buffer)
     - Pushes grid dimensions via push constants
     - Draws instanced quads (6 vertices × N×M instances)
   - **Renders particles on top**:
     - Binds particle pipeline
     - Binds descriptor set (with position buffer)
     - Draws N points (where N = number of particles)
   - Ends render pass
3. **Present** the image to the screen

## Integration Example

The graphics renderer integrates with the compute pipeline by:
1. Sharing the same Vulkan device and queue between compute and graphics
2. Using `set_grid_buffer()` to bind the GridCell buffer from compute
3. Using `set_position_buffer()` to bind the particle position buffer from compute
4. Calling `render_frame()` in the window event loop to present each frame

See `src/main.rs` for a complete example of the integration.

## Shader Details

### Grid Rendering Shaders
- **`grid_vs` (Vertex Shader)**: Uses instanced rendering to generate quads procedurally. Each instance represents one grid cell. The shader calculates the cell position from `instance_index`, generates 6 vertices per quad based on `vertex_index`, reads mass from the GridCell buffer, and passes it to the fragment shader.
- **`grid_fs` (Fragment Shader)**: Converts mass values [0, 1] to grayscale colors for the heatmap visualization.

### Particle Rendering Shaders
- **`main_vs` (Vertex Shader)**: Reads Vec2 positions from storage buffer using `vertex_index`, converts from [0, 1] normalized space to [-1, 1] clip space.
- **`main_fs` (Fragment Shader)**: Outputs white color for all particles.

## Performance Considerations

- **Simple synchronization**: Current implementation waits for each frame to complete before starting the next. For better performance, you'd want to implement proper fence management per swapchain image.
  
- **Instanced rendering**: The grid uses instanced drawing which is efficient for large numbers of cells. Each instance generates 6 vertices (2 triangles) for a quad.

- **Point size**: Each particle is rendered as a 1-pixel point. For larger/variable sizes, you'd need to use a geometry shader or instanced quads.

- **Storage buffers**: Using storage buffers instead of vertex buffers has some overhead but provides more flexibility and enables direct GPU-to-GPU data sharing between compute and graphics.

- **Command buffer recreation**: Currently recreates command buffers when buffers are updated. For better performance, could use multiple descriptor sets and switch between them.

## Technical Details

### Push Constants
The grid pipeline uses push constants (`GridPushConstants` in `shared/src/grid.rs`) to pass grid dimensions. Push constants are a lightweight way to pass small amounts of data (< 128 bytes) to shaders without using descriptor sets.

### Descriptor Set Layout
- **Set 0, Binding 0**: Particle position buffer (`Vec2[]`)
- **Set 0, Binding 1**: Grid cell buffer (`GridCell[]`)

Each pipeline only binds the descriptor set with its relevant buffer.

### Render Order
The command buffer renders in this order:
1. Grid pipeline (background)
2. Particle pipeline (foreground)

This ensures particles appear on top of the heatmap.

## Future Enhancements

Possible improvements:
1. **Better synchronization**: Track fences per swapchain image for proper frame pacing
2. **Variable point sizes**: Use geometry shader to expand points to quads
3. **Colorful heatmap**: Replace grayscale with a color gradient (e.g., blue→green→red)
4. **More visual effects**: Add trails, colors based on velocity, etc.
5. **Multiple render passes**: Implement post-processing effects
6. **Dynamic grid resolution**: Allow runtime changes to grid dimensions
7. **Velocity visualization**: Render velocity vectors as oriented quads
