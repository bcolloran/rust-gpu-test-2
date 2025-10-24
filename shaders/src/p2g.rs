use core::u32;

use glam::UVec3;
use shared::{
    grid::{linear_grid_index, STENCIL_OFFSETS_POS},
    INV_DX, N_GRID_TOTAL, P_MASS,
};
use spirv_std::{
    arch::atomic_f_add,
    glam::{self, Vec2},
    spirv,
};

use spirv_std::memory::{Scope, Semantics};

const SCOPE: u32 = Scope::Device as u32;
const SEMANTICS: u32 = Semantics::NONE.bits();

#[spirv(compute(threads(64)))]
pub fn p2g(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] xs: &mut [Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] vs: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] grid: &mut [shared::grid::GridCell],
) {
    let p = id.x as usize;

    let xp = xs[p];

    // let index = grid_index_unit_xy(xp.x, xp.y);

    let base = (xp * INV_DX - 0.5).floor();
    let base_idx = base.as_uvec2();
    let fx = xp * INV_DX - base;

    let w = [
        0.5 * (1.5 - fx) * (1.5 - fx),
        0.75 - (fx - 1.0) * (fx - 1.0),
        0.5 * (fx - 0.5) * (fx - 0.5),
    ];

    for o in 0..9 {
        let offset = STENCIL_OFFSETS_POS[o];
        // let dpos = (offset.as_vec2() - fx) * DX;

        let weight = w[offset.x as usize].x * w[offset.y as usize].y;

        let grid_x = base_idx.x + offset.x as u32;
        let grid_y = base_idx.y + offset.y as u32;
        let index = linear_grid_index(grid_x, grid_y);

        if index >= N_GRID_TOTAL as usize {
            continue;
        }

        let m = &mut grid[index].mass;
        let mass_add = weight * P_MASS;
        unsafe { atomic_f_add::<_, SCOPE, SEMANTICS>(m, mass_add) };
    }

    // let mass = 10.0;
    // let velocity = vec2(1.0, -9.0);
    // let m = &mut grid[index].mass;
}
