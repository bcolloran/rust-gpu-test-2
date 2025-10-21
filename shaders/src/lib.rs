#![cfg_attr(target_arch = "spirv", no_std)]
// HACK(eddyb) can't easily see warnings otherwise from `spirv-builder` builds.
// #![deny(warnings)]

use core::u32;

use glam::UVec3;
use spirv_std::{
    glam::{self, vec2, Vec2, Vec4},
    spirv,
};

pub mod bindless;
pub mod mult;

#[spirv(compute(threads(64)))]
pub fn adder(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] b: &[u32],
) {
    let i = id.x as usize;
    shared::add_update(&mut a[i], b[i]);
}

#[spirv(compute(threads(64)))]
pub fn step_particles(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] x: &mut [Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] v: &[Vec2],
) {
    let i = id.x as usize;
    shared::add_update(&mut x[i], v[i]);
}

#[spirv(compute(threads(64)))]
pub fn wrap_particles(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] x: &mut [Vec2],
) {
    let i = id.x as usize;
    x[i] = x[i] % Vec2::splat(1.0);
}

#[inline]
pub fn wang32(mut x: u32) -> u32 {
    x = x.wrapping_add(!x << 15);
    x ^= x >> 10;
    x = x.wrapping_add(x << 3);
    x ^= x >> 6;
    x = x.wrapping_add(!x << 11);
    x ^ (x >> 16)
}

#[inline]
pub fn hash_many<const N: usize>(xs: [u32; N]) -> u32 {
    // seed
    let mut acc: u32 = 0x9E37_79B9;

    // FIXME use iterator
    for i in 0..N {
        let y = xs[i].wrapping_mul(0x9E37_79B9) ^ (xs[i] >> 16);
        acc ^= y.wrapping_add(0x85EB_CA6B);
        acc = acc.rotate_left(13);
    }
    wang32(acc) // or quick_mix32(acc)
}

#[inline]
fn rand_f32<const N: usize>(xs: [u32; N]) -> f32 {
    (hash_many(xs) as f32) / (u32::MAX as f32)
}

#[spirv(compute(threads(16, 16)))]
pub fn fill_grid_random(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] grid: &mut [shared::grid::GridCell],
) {
    let grid_width = 256; // Example fixed grid width
    let x = id.x;
    let y = id.y;
    let index = y as usize * grid_width + x as usize;

    // Simple pseudo-random generation based on indices
    let mass = rand_f32([x, y, 0]);
    let velocity = vec2(rand_f32([x, y, 2]), rand_f32([x, y, 4]));

    // let mass = 10.0;
    // let velocity = vec2(1.0, -9.0);

    grid[index].mass = mass;
    grid[index].velocity = velocity;
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
