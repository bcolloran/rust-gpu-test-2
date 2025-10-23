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
