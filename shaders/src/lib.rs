#![cfg_attr(target_arch = "spirv", no_std)]
// HACK(eddyb) can't easily see warnings otherwise from `spirv-builder` builds.
// #![deny(warnings)]

pub mod bindless;
pub mod mult;
pub mod p2g;
pub mod render;

use core::u32;

pub use render::{
    grid_density::{grid_density_fs, grid_density_vs},
    particles::{particles_fs, particles_vs},
};

use glam::UVec3;
use shared::grid::{linear_grid_index, linear_grid_index_unit_xy};
use spirv_std::{
    arch::atomic_f_add,
    glam::{self, vec2, Vec2},
    spirv,
};

use spirv_std::memory::{Scope, Semantics};

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
    let x = id.x;
    let y = id.y;
    let index = linear_grid_index(x, y);

    // Simple pseudo-random generation based on indices
    let mass = rand_f32([x, y, 0]);
    let velocity = vec2(rand_f32([x, y, 2]), rand_f32([x, y, 4]));

    // let mass = 10.0;
    // let velocity = vec2(1.0, -9.0);

    grid[index].mass = mass;
    grid[index].v = velocity;
}

#[spirv(compute(threads(8, 8)))]
pub fn clear_grid(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] grid: &mut [shared::grid::GridCell],
) {
    let x = id.x;
    let y = id.y;
    let index = linear_grid_index(x, y);

    grid[index].mass = 0.0;
    grid[index].v = Vec2::ZERO;
}

#[spirv(compute(threads(64)))]
pub fn p2g_simple_test(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] ps: &mut [Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] grid: &mut [shared::grid::GridCell],
) {
    let i = id.x as usize;

    let p = ps[i];

    let index = linear_grid_index_unit_xy(p.x, p.y);

    // let mass = 10.0;
    // let velocity = vec2(1.0, -9.0);
    let m = &mut grid[index].mass;

    const SCOPE: u32 = Scope::Device as u32;
    const SEMANTICS: u32 = Semantics::NONE.bits();
    unsafe { atomic_f_add::<_, SCOPE, SEMANTICS>(m, 0.1) };
}
