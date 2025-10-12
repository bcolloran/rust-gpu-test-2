#![cfg_attr(target_arch = "spirv", no_std)]
// HACK(eddyb) can't easily see warnings otherwise from `spirv-builder` builds.
#![deny(warnings)]

use core::ops::{Add, AddAssign};

use glam::UVec3;
use spirv_std::{
    glam::{self, vec2, Vec2, Vec4},
    spirv,
};

pub mod mult;

// Adapted from the wgpu hello-compute example

/// Returns the length of the Collatz sequence (excluding the starting number) for `n`. Returns
/// `None` if (a) `n` is zero, or (b) a number in the sequence overflows a `u32`.
///
/// # Examples
///
/// The sequence for 3 (excluding the starting number) is `[10, 5, 16, 8, 4, 2, 1]`, which has
/// length 7.
/// ```
/// # use compute_shader::collatz;
/// assert_eq!(collatz(3), Some(7));
/// ```
pub fn collatz(mut n: u32) -> Option<u32> {
    let mut i = 0;
    if n == 0 {
        return None;
    }
    while n != 1 {
        n = if n % 2 == 0 {
            n / 2
        } else {
            // Overflow? (i.e. 3*n + 1 > 0xffff_ffff)
            if n >= 0x5555_5555 {
                return None;
            }
            // TODO: Use this instead when/if checked add/mul can work:
            // n.checked_mul(3)?.checked_add(1)?
            3 * n + 1
        };
        i += 1;
    }
    Some(i)
}

// LocalSize/numthreads of (x = 64, y = 1, z = 1)
#[spirv(compute(threads(64)))]
pub fn main_cs(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] prime_indices: &mut [u32],
) {
    let index = id.x as usize;
    prime_indices[index] = collatz(prime_indices[index]).unwrap_or(u32::MAX);
}

fn add_update<T: Add + AddAssign>(a: &mut T, b: T) {
    *a += b
}

#[spirv(compute(threads(64)))]
pub fn adder(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] b: &[u32],
) {
    let i = id.x as usize;
    add_update(&mut a[i], b[i]);
}

#[spirv(compute(threads(64)))]
pub fn step_particles(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] x: &mut [Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] v: &[Vec2],
) {
    let i = id.x as usize;
    add_update(&mut x[i], v[i]);
}

#[spirv(compute(threads(64)))]
pub fn wrap_particles(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] x: &mut [Vec2],
) {
    let i = id.x as usize;
    x[i] = x[i] % Vec2::splat(1.0);
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
    out_color: &mut Vec4,
) {
    let idx = vert_idx as usize;

    // Read position from buffer (assumed to be in [0, 1] range)
    let pos = positions[idx];

    // Convert from [0, 1] to [-1, 1] (Vulkan clip space)
    // Note: Vulkan has Y axis pointing down, so we might want to flip Y
    let clip_pos = pos * 2.0 - Vec2::ONE;

    *builtin_pos = clip_pos.extend(0.0).extend(1.0);

    // Generate a simple color based on the vertex index for visual variety
    // We use modulo and simple arithmetic to create color variation
    let color_factor = (idx % 256) as f32 / 256.0;
    *out_color = Vec4::new(
        0.3 + 0.7 * color_factor,         // Red varies from 0.3 to 1.0
        0.5 + 0.5 * (1.0 - color_factor), // Green varies inversely
        0.8,                              // Blue is constant
        1.0,                              // Full opacity
    );
}

/// Fragment shader for coloring the rendered points
///
/// This simply outputs the interpolated color from the vertex shader.
#[spirv(fragment)]
pub fn main_fs(in_color: Vec4, output: &mut Vec4) {
    *output = in_color;
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
