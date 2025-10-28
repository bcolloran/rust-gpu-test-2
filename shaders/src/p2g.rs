// use core::u32;

use glam::UVec3;
use shared::{
    grid::{linear_grid_index_ivec_unchecked, STENCIL_OFFSETS},
    mpm_utils::quadratic_weight_2d,
    particles::{Material, MaterialPod, ParticleDeformation, ParticleMatrices},
    DT, DX, INV_DX, LAMBDA_0, MU_0, N_GRID_X, P_MASS,
};
use spirv_std::{
    arch::atomic_f_add,
    glam::{self, vec2, Mat2, Vec2},
    num_traits::float::Float,
    spirv,
};

use spirv_std::memory::{Scope, Semantics};

use crate::{
    svd::{svd2x2_exact, Svd2},
    util::atomic_f_add_vec2,
};

const SCOPE: u32 = Scope::Device as u32;
const SEMANTICS: u32 = Semantics::NONE.bits();

#[allow(non_snake_case)]
#[spirv(compute(threads(64)))]
pub fn p2g(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] xs: &mut [Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] vs: &[Vec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] grid: &mut [shared::grid::GridCell],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 5)]
    particle_matrices: &mut [ParticleMatrices],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 6)]
    particle_deformation: &mut [ParticleDeformation],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 7)]
    particle_material: &mut [MaterialPod],
) {
    let p = id.x as usize;
    let xp = xs[p];
    let vp = vs[p];
    let C = particle_matrices[p].C;
    let mut F = particle_matrices[p].F;
    let mut Jp = particle_deformation[p].J;
    let material = particle_material[p].to_material();

    // update deformation gradient F
    F = Mat2::IDENTITY + DT * C.mul_mat2(&F);
    // hardening coefficient
    let h = (10.0 * (1.0 - particle_deformation[p].J))
        .exp()
        .clamp(0.1, 5.0);

    let (mu, lambda) = match material {
        Material::Fluid => (0.0, 0.0),
        // jelly-like solid, less stiff; h=0.3
        Material::Jelly => (MU_0 * 0.3, LAMBDA_0 * 0.3),
        Material::Snow => (MU_0 * h, LAMBDA_0 * h),
    };

    let Svd2 {
        u: U,
        s: mut sig,
        v: V,
    } = svd2x2_exact(F.clone());
    let mut J = 1.0;
    for d in 0..2 {
        let mut new_sig = sig[d];
        if matches!(material, Material::Snow) {
            // plasticity for snow
            new_sig = new_sig.clamp(1.0 - 2.5e-2, 1.0 + 4.5e-3);
        }
        Jp *= sig[d] / new_sig;
        sig[d] = new_sig;
        J *= new_sig;
    }
    match material {
        Material::Fluid => {
            F = Mat2::IDENTITY * J.sqrt();
        }
        Material::Snow | Material::Jelly => {
            // Reconstruct F
            F = U
                .mul_mat2(&Mat2::from_diagonal(sig))
                .mul_mat2(&V.transpose());
        }
    }
    let stress = 2.0 * mu * (F - U.mul_mat2(&V.transpose())) * F.transpose()
        + Mat2::IDENTITY * lambda * (J - 1.0) * J;
    let affine_stress = stress + P_MASS * C;

    // save updated particle data
    particle_matrices[p].F = F;
    particle_deformation[p].J = Jp;

    // do the actual P2G transfer
    let containing_cell = (xp * INV_DX).floor();
    // index of the cell containing the particle (usting graphics coords from top-left)
    let containing_idx = containing_cell.as_ivec2();
    let containing_center = (containing_cell + vec2(0.5, 0.5)) * DX;
    for o in 0..9 {
        let offset = STENCIL_OFFSETS[o];
        let grid_idx = containing_idx + offset;
        if grid_idx.x < 0
            || grid_idx.y < 0
            || grid_idx.x >= N_GRID_X as i32
            || grid_idx.y >= N_GRID_X as i32
        {
            continue;
        }
        let index = unsafe {
            // Safety: bounds just checked above
            linear_grid_index_ivec_unchecked(grid_idx)
        };

        let grid_pos = containing_center + (offset.as_vec2()) * DX;

        let weight = quadratic_weight_2d(xp - grid_pos);

        let m = &mut grid[index].mass;
        let v = &mut grid[index].v;
        let mass_add = weight * P_MASS;
        let v_add = weight * (P_MASS * vp + affine_stress.mul_vec2(grid_pos));
        unsafe { atomic_f_add::<_, SCOPE, SEMANTICS>(m, mass_add) };
        unsafe { atomic_f_add_vec2::<SCOPE, SEMANTICS>(v, v_add) };
    }
}
