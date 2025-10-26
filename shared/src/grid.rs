use core::clone::Clone;

use bytemuck::{Pod, Zeroable};
use spirv_std::glam::{IVec2, UVec2, Vec2};

use crate::N_GRID_X;

#[allow(non_snake_case)]
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GridCell {
    #[zeroable]
    #[bytemuck]
    pub v: Vec2,
    #[zeroable]
    pub mass: f32,
}

#[inline(always)]
pub fn linear_grid_index(x: u32, y: u32) -> usize {
    (y * N_GRID_X + x) as usize
}

#[inline(always)]
pub fn linear_grid_index_uvec(idx: UVec2) -> usize {
    (idx.y * N_GRID_X + idx.x) as usize
}

/// compute the linear index in a GRID_SIZE x GRID_SIZE grid from (x, y) in [0.0, 1.0] range
#[inline(always)]
pub fn linear_grid_index_unit_xy(x: f32, y: f32) -> usize {
    let grid_x = (x * N_GRID_X as f32) as u32;
    let grid_y = (y * N_GRID_X as f32) as u32;
    linear_grid_index(grid_x, grid_y)
}

/// Compute the 2d grid index in a GRID_SIZE x GRID_SIZE grid from (x, y) in [0.0, 1.0] range
#[inline(always)]
pub fn grid_index_unit_xy(x: f32, y: f32) -> UVec2 {
    let grid_x = (x * N_GRID_X as f32) as u32;
    let grid_y = (y * N_GRID_X as f32) as u32;
    UVec2::new(grid_x, grid_y)
}

/// G

/// Offsets for a 3x3 stencil around a grid cell.
pub const STENCIL_OFFSETS: [IVec2; 9] = [
    IVec2::new(-1, -1),
    IVec2::new(0, -1),
    IVec2::new(1, -1),
    IVec2::new(-1, 0),
    IVec2::new(0, 0),
    IVec2::new(1, 0),
    IVec2::new(-1, 1),
    IVec2::new(0, 1),
    IVec2::new(1, 1),
];

/// Offsets for a 3x3 stencil around a grid cell.
/// Positive version, useful for indexing.
pub const STENCIL_OFFSETS_POS: [UVec2; 9] = [
    UVec2::new(0, 0),
    UVec2::new(1, 0),
    UVec2::new(2, 0),
    UVec2::new(0, 1),
    UVec2::new(1, 1),
    UVec2::new(2, 1),
    UVec2::new(0, 2),
    UVec2::new(1, 2),
    UVec2::new(2, 2),
];

/// Push constants structure for grid rendering
///
/// This is used to pass the grid dimensions to the vertex shader.
/// Push constants are a lightweight way to pass small amounts of data to shaders.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GridPushConstants {
    pub grid_width: u32,
    pub grid_height: u32,
}
