#![cfg_attr(target_arch = "spirv", no_std)]
// HACK(eddyb) can't easily see warnings otherwise from `spirv-builder` builds.
// #![deny(warnings)]

use core::u32;

use glam::UVec3;
use shared::grid::{grid_index, grid_index_unit_xy, GRID_SIZE};
use spirv_std::{
    arch::atomic_f_add,
    glam::{self, vec2, Vec2, Vec4},
    num_traits::float::FloatCore,
    spirv,
};

use spirv_std::memory::{Scope, Semantics};

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
    if x[i].x < 0.0 {
        x[i].x += 1.0;
    }
    if x[i].x >= 1.0 {
        x[i].x -= 1.0;
    }
    if x[i].y < 0.0 {
        x[i].y += 1.0;
    }
    if x[i].y >= 1.0 {
        x[i].y -= 1.0;
    }
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
    wang32(acc)
}

#[inline]
fn rand_f32<const N: usize>(xs: [u32; N]) -> f32 {
    (hash_many(xs) as f32) / (u32::MAX as f32)
}

#[spirv(compute(threads(8, 8)))]
pub fn fill_grid_random(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] grid: &mut [shared::grid::GridCell],
) {
    let grid_width = 256; // Example fixed grid width
    let x = id.x;
    let y = id.y;
    let index = grid_index(x, y);

    // Simple pseudo-random generation based on indices
    let mass = rand_f32([x, y, 0]);
    let velocity = vec2(rand_f32([x, y, 2]), rand_f32([x, y, 4]));

    // let mass = 10.0;
    // let velocity = vec2(1.0, -9.0);

    grid[index].mass = mass;
    grid[index].velocity = velocity;
}

#[spirv(compute(threads(8, 8)))]
pub fn clear_grid_mass(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] grid: &mut [shared::grid::GridCell],
) {
    let x = id.x;
    let y = id.y;
    let index = grid_index(x, y);

    grid[index].mass = 0.0;
}

#[spirv(compute(threads(64)))]
pub fn p2g(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] ps: &mut [Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] grid: &mut [shared::grid::GridCell],
) {
    let i = id.x as usize;

    let p = ps[i];

    let index = grid_index_unit_xy(p.x, p.y);

    // let mass = 10.0;
    // let velocity = vec2(1.0, -9.0);
    let m = &mut grid[index].mass;

    const SCOPE: u32 = Scope::Device as u32;
    const SEMANTICS: u32 = Semantics::NONE.bits();
    unsafe { atomic_f_add::<_, SCOPE, SEMANTICS>(m, 1.0) };
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

// ==============================================================================
// GRID RENDERING SHADERS - For rendering the grid heatmap
// ==============================================================================

/// Vertex shader for rendering the grid as a heatmap
///
/// This shader renders the grid using instanced quads. Each instance represents
/// one grid cell, and we generate a quad (2 triangles = 6 vertices) for each cell.
///
/// The shader:
/// 1. Uses instance_index to determine which grid cell this is
/// 2. Uses vertex_index (0-5) to determine which corner of the quad
/// 3. Reads the GridCell data (mass, velocity) from the storage buffer
/// 4. Positions the quad to cover the appropriate screen region
/// 5. Passes the mass value to the fragment shader for coloring
///
/// The grid is rendered behind the particles (drawn first in the command buffer).
#[spirv(vertex)]
pub fn grid_vs(
    #[spirv(vertex_index)] vert_idx: i32,
    #[spirv(instance_index)] inst_idx: i32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] grid: &[shared::grid::GridCell],
    #[spirv(push_constant)] push_constants: &shared::grid::GridPushConstants,
    #[spirv(position)] builtin_pos: &mut Vec4,
    out_mass: &mut f32,
) {
    let grid_width = push_constants.grid_width;
    let grid_height = push_constants.grid_height;

    // Calculate row and column from instance index
    let col = (inst_idx as u32) % grid_width;
    let row = (inst_idx as u32) / grid_width;

    // Read grid cell data
    let cell = grid[inst_idx as usize];
    let mass = cell.mass;

    // Calculate cell size in normalized coordinates [0, 1]
    let cell_width = 1.0 / (grid_width as f32);
    let cell_height = 1.0 / (grid_height as f32);

    // Calculate cell position in normalized coordinates [0, 1]
    let cell_x = (col as f32) * cell_width;
    let cell_y = (row as f32) * cell_height;

    // Generate quad vertices based on vertex_index
    // We use a triangle list with 6 vertices per quad:
    // 0,1,2 for first triangle, 3,4,5 for second triangle
    // Layout:
    //   0 --- 1/3
    //   |  \   |
    //   2/4 -- 5
    let local_vert_idx = vert_idx % 6;
    let (dx, dy) = match local_vert_idx {
        0 => (0.0, 0.0), // top-left
        1 => (1.0, 0.0), // top-right
        2 => (0.0, 1.0), // bottom-left
        3 => (1.0, 0.0), // top-right (second triangle)
        4 => (0.0, 1.0), // bottom-left (second triangle)
        5 => (1.0, 1.0), // bottom-right
        _ => (0.0, 0.0), // should never happen
    };

    // Calculate final position in [0, 1] space
    let pos_x = cell_x + dx * cell_width;
    let pos_y = cell_y + dy * cell_height;

    // Convert from [0, 1] to [-1, 1] clip space
    let clip_x = pos_x * 2.0 - 1.0;
    let clip_y = pos_y * 2.0 - 1.0;

    *builtin_pos = Vec4::new(clip_x, clip_y, 0.0, 1.0);
    *out_mass = mass;
}

/// Fragment shader for rendering the grid heatmap
///
/// This shader converts the mass value (assumed to be in [0, 1]) to a grayscale color.
/// Higher mass values appear brighter (whiter), lower values appear darker (blacker).
#[spirv(fragment)]
pub fn grid_fs(in_mass: f32, output: &mut Vec4) {
    // Clamp mass to [0, 1] range to be safe
    let mass_clamped = 0.25 * in_mass.clamp(0.0, 1.0);

    // Simple grayscale mapping: mass directly maps to brightness
    // You could add color mapping here for a more interesting heatmap
    *output = Vec4::new(mass_clamped, mass_clamped, mass_clamped, 1.0);
}
