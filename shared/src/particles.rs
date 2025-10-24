use core::clone::Clone;

use bytemuck::{Pod, Zeroable};
use spirv_std::glam::Mat2;

#[allow(non_snake_case)]
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GridCellMats {
    #[zeroable]
    #[bytemuck]
    pub C: Mat2,
    #[zeroable]
    #[bytemuck]
    pub F: Mat2,
}

#[allow(non_snake_case)]
#[repr(u8)]
#[derive(Copy, Clone, Debug, Zeroable)]
pub enum Material {
    Fluid = 0,
    Solid = 1,
    Snow = 2,
}
