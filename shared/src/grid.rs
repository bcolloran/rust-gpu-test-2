use core::clone::Clone;

use bytemuck::{Pod, Zeroable};
use spirv_std::glam::Vec2;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GridCell {
    #[zeroable]
    pub mass: f32,
    #[zeroable]
    #[bytemuck]
    pub velocity: Vec2,
}

pub const GRID_SIZE: u32 = 16;

pub const fn grid_index(x: u32, y: u32) -> usize {
    (y * GRID_SIZE + x) as usize
}

/// compute the linear index in a GRID_SIZE x GRID_SIZE grid from (x, y) in [0.0, 1.0] range
pub const fn grid_index_unit_xy(x: f32, y: f32) -> usize {
    let grid_x = (x * GRID_SIZE as f32) as u32;
    let grid_y = (y * GRID_SIZE as f32) as u32;
    grid_index(grid_x, grid_y)
}

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
