use glam::UVec3;
use spirv_std::{
    glam::{self, vec2, Vec2, Vec4},
    spirv,
};

// use crate::add_update;
// ==============================================================================
// BINDLESS COMPUTE SHADERS
// ==============================================================================
//
// In a bindless approach, instead of binding each logical buffer to a separate
// descriptor binding slot, we use a smaller number of large buffers that contain
// multiple logical buffers packed together. The shader uses push constants to
// receive offsets that tell it where each logical buffer starts within the large buffer.
//
// This approach has several advantages:
// 1. Fewer descriptor set bindings needed (2 bindings instead of 6+)
// 2. Fewer descriptor set updates/rebindings needed between dispatches
// 3. More flexible - can easily change which logical buffers are used without
//    recreating descriptor sets
// 4. Better utilization of GPU memory and descriptor limits
//
// The main tradeoff is slightly more complex shader code, as we need to calculate
// offsets. However, push constants are very fast to update, so this is typically
// a worthwhile trade.

/// Push constants for the adder shader
/// These tell the shader where in the unified buffer each logical buffer starts
#[repr(C)]
pub struct AdderPushConstants {
    pub a_offset: u32,    // Offset (in elements) where buffer 'a' starts
    pub b_offset: u32,    // Offset (in elements) where buffer 'b' starts
    pub buffer_size: u32, // Size of each logical buffer (number of elements to process)
    pub _padding: u32,    // Padding to align to 16 bytes (required by Vulkan)
}

/// Push constants for particle shaders (step and wrap)
#[repr(C)]
pub struct ParticlePushConstants {
    pub x_offset: u32,    // Offset (in elements) where buffer 'x' starts
    pub v_offset: u32,    // Offset (in elements) where buffer 'v' starts (unused for wrap)
    pub buffer_size: u32, // Size of each logical buffer (number of elements to process)
    pub _padding: u32,    // Padding to align to 16 bytes
}

/// Bindless adder shader
///
/// Instead of having 'a' and 'b' as separate bindings, this shader receives:
/// - A single large unified buffer containing all u32 data (binding = 0)
/// - Push constants telling it the offsets where 'a' and 'b' start in that buffer
///
/// This means we can invoke the shader many times with different a/b combinations
/// without rebinding descriptor sets - we just update the push constants.
///
/// IMPORTANT: Since we dispatch with a fixed workgroup size (64 threads), but may have
/// fewer elements, we need bounds checking to prevent out-of-bounds access.
#[spirv(compute(threads(64)))]
pub fn adder(
    #[spirv(global_invocation_id)] id: UVec3,
    // BINDLESS: Single large buffer containing all u32 data
    // All logical buffers (a, b, c, d, etc.) are packed into this one buffer
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] unified_u32_buffer: &mut [u32],
    // Push constants tell us where each logical buffer starts
    #[spirv(push_constant)] push_constants: &AdderPushConstants,
) {
    let i = id.x as usize;
    let a_offset = push_constants.a_offset as usize;
    let b_offset = push_constants.b_offset as usize;
    let buffer_size = push_constants.buffer_size as usize;

    // Bounds check: only process if this thread's index is within the logical buffer size
    // This is crucial because we dispatch with a fixed workgroup size (64 threads),
    // but the actual data may be smaller (e.g., 8 elements)
    if i < buffer_size {
        // SAFETY: We have already checked that i is within buffer_size,
        // let [a, b] =
        //     unsafe { unified_u32_buffer.get_disjoint_unchecked_mut([a_offset + i, b_offset + i]) };

        // *a += *b;
        // add_update(a, *b);
        unified_u32_buffer[a_offset + i] += unified_u32_buffer[b_offset + i];
    }
}

/// Bindless step_particles shader
///
/// Similar to adder, but operates on Vec2 data (positions and velocities)
#[spirv(compute(threads(64)))]
pub fn step_particles(
    #[spirv(global_invocation_id)] id: UVec3,
    // BINDLESS: Single large buffer containing all Vec2 data
    // Both position buffer 'x' and velocity buffer 'v' are packed into this one buffer
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] unified_vec2_buffer: &mut [Vec2],
    #[spirv(push_constant)] push_constants: &ParticlePushConstants,
) {
    let i = id.x as usize;
    let x_offset = push_constants.x_offset as usize;
    let v_offset = push_constants.v_offset as usize;
    let buffer_size = push_constants.buffer_size as usize;

    // Bounds check to prevent out-of-bounds access from extra threads
    if i < buffer_size {
        // x[i] += v[i] becomes:
        unified_vec2_buffer[x_offset + i] += unified_vec2_buffer[v_offset + i];
    }
}

/// Bindless wrap_particles shader
///
/// Applies modulo wrapping to particle positions
#[spirv(compute(threads(64)))]
pub fn wrap_particles(
    #[spirv(global_invocation_id)] id: UVec3,
    // BINDLESS: Single large buffer containing all Vec2 data
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] unified_vec2_buffer: &mut [Vec2],
    #[spirv(push_constant)] push_constants: &ParticlePushConstants,
) {
    let i = id.x as usize;
    let x_offset = push_constants.x_offset as usize;
    let buffer_size = push_constants.buffer_size as usize;

    // Bounds check to prevent out-of-bounds access from extra threads
    if i < buffer_size {
        // x[i] = x[i] % 1.0 becomes:
        unified_vec2_buffer[x_offset + i] = unified_vec2_buffer[x_offset + i] % Vec2::splat(1.0);
    }
}

// ==============================================================================
// GRAPHICS SHADERS - For rendering points to the screen
// ==============================================================================

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
pub fn main_vs(
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
pub fn main_fs(output: &mut Vec4) {
    // Simple white color
    *output = Vec4::new(1.0, 1.0, 1.0, 1.0);
}

// ==============================================================================
// ALTERNATIVE: Full-screen triangle shaders (keep for reference)
// ==============================================================================

/// Alternative vertex shader that creates a full-screen triangle
/// This doesn't need any vertex buffer - it generates positions from the vertex index
#[spirv(vertex)]
pub fn fullscreen_vs(
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
pub fn fullscreen_fs(#[spirv(frag_coord)] in_frag_coord: Vec4, output: &mut Vec4) {
    *output = in_frag_coord / Vec4::new(800.0, 600.0, 1.0, 1.0);
}
