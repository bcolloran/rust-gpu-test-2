use core::clone::Clone;

use bytemuck::{Pod, Zeroable};
use spirv_std::glam::Vec2;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Cell {
    #[zeroable]
    pub mass: f32,
    #[zeroable]
    #[bytemuck]
    pub velocity: Vec2,
}
