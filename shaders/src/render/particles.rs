use spirv_std::{
    glam::{vec2, Vec2, Vec4},
    spirv,
};

/// Vertex shader for rendering the particle positions
///
/// This shader renders individual points by reading from a storage buffer containing Vec2 positions.
/// Each Vec2 represents a point in normalized coordinates [0, 1] x [0, 1].
///
/// The vertex shader:
/// 1. Uses the vertex_index to look up the position from the buffer
/// 2. Converts from [0, 1] range to Vulkan's clip space [-1, 1]
/// 3. Outputs the position and passes through a color based on vertex index
#[spirv(vertex)]
pub fn particles_vs(
    #[spirv(vertex_index)] vert_idx: i32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] positions: &[Vec2],
    #[spirv(position)] builtin_pos: &mut Vec4,
) {
    let idx = vert_idx as usize;

    // Read position from buffer (assumed to be in [0, 1] range)
    let pos = positions[idx];

    // Convert from [0, 1] to [-1, 1] (Vulkan clip space)
    // Note: Vulkan has Y axis pointing down, so we might want to flip Y
    let clip_pos = pos * 2.0 - Vec2::ONE;

    *builtin_pos = clip_pos.extend(0.0).extend(1.0);
}

/// Fragment shader for coloring the rendered points
///
/// This outputs a simple constant color for all pixels.
#[spirv(fragment)]
pub fn particles_fs(output: &mut Vec4) {
    // Simple white color
    *output = Vec4::new(1.0, 1.0, 1.0, 1.0);
}

// ============
// ALTERNATIVE: Full-screen triangle shaders (keep for reference)

/// Alternative vertex shader that creates a full-screen triangle
/// This doesn't need any vertex buffer - it generates positions from the vertex index
#[spirv(vertex)]
pub fn particles_fullscreen_vs(
    #[spirv(vertex_index)] vert_idx: i32,
    #[spirv(position)] builtin_pos: &mut Vec4,
) {
    // Create a "full screen triangle" by mapping the vertex index.
    // ported from https://www.saschawillems.de/blog/2016/08/13/vulkan-tutorial-on-rendering-a-fullscreen-quad-without-buffers/
    let uv = vec2(((vert_idx << 1) & 2) as f32, (vert_idx & 2) as f32);
    let pos = 2.0 * uv - Vec2::ONE;

    *builtin_pos = pos.extend(0.0).extend(1.0);
}

/// Alternative fragment shader that colors based on screen position
#[spirv(fragment)]
pub fn particles_fullscreen_fs(#[spirv(frag_coord)] in_frag_coord: Vec4, output: &mut Vec4) {
    *output = in_frag_coord / Vec4::new(2000.0, 2000.0, 1.0, 1.0);
}
