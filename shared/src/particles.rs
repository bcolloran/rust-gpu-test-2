use core::clone::Clone;

use bytemuck::{Pod, Zeroable};
use spirv_std::glam::Mat2;

#[allow(non_snake_case)]
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ParticleMatrices {
    #[zeroable]
    #[bytemuck]
    pub C: Mat2,
    #[zeroable]
    #[bytemuck]
    pub F: Mat2,
}
impl ParticleMatrices {
    #[inline]
    pub fn new() -> Self {
        Self {
            C: Mat2::ZERO,
            F: Mat2::IDENTITY,
        }
    }
}

#[allow(non_snake_case)]
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ParticleDeformation {
    #[zeroable]
    #[bytemuck]
    /// plastic deformation determinant
    pub J: f32,
}
impl ParticleDeformation {
    #[inline]
    pub fn new() -> Self {
        Self { J: 1.0 }
    }
}

#[allow(non_snake_case)]
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub enum Material {
    Fluid = 0,
    Solid = 1,
    Snow = 2,
}

impl From<u8> for Material {
    fn from(value: u8) -> Self {
        match value {
            0 => Material::Fluid,
            1 => Material::Solid,
            2 => Material::Snow,
            _ => panic!("Invalid material value: {}", value),
        }
    }
}

impl From<u32> for Material {
    fn from(value: u32) -> Self {
        (value as u8).into()
    }
}

// Bridge type: POD representation of the enum’s C integer.
#[repr(C)]
#[derive(Copy, Clone, Debug, Zeroable, Pod)]
pub struct MaterialPod(u8);
impl MaterialPod {
    #[inline]
    pub fn to_material(&self) -> Material {
        match self.0 {
            0 => Material::Fluid,
            1 => Material::Solid,
            2 => Material::Snow,
            _ => panic!("Invalid material value: {}", self.0),
        }
    }
    #[inline]
    pub fn u8(&self) -> u8 {
        self.0
    }
}

// Conversions
impl From<Material> for MaterialPod {
    fn from(m: Material) -> Self {
        Self(m as u8)
    }
}
impl From<MaterialPod> for u8 {
    fn from(mp: MaterialPod) -> Self {
        mp.0
    }
}
