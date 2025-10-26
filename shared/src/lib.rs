//! Shared types for the compute demos
#![no_std]

pub mod grid;
pub mod particles;

pub struct RowA {
    pub x: u32,
    pub y: f32,
}

use bytemuck::{Pod, Zeroable};
use core::fmt::{self, Display};
use core::ops::{Add, AddAssign};

pub fn add_update<T: Add + AddAssign>(a: &mut T, b: T) {
    *a += b
}

pub const QUALITY: u32 = 1;
pub const N_PARTICLES: u32 = 9 * QUALITY * QUALITY;
pub const MATERIAL_GROUP_SIZE: u32 = N_PARTICLES / 3;

// grid constants

/// Number of grid cells along one dimension
pub const N_GRID: u32 = 128 * QUALITY;
/// total number of grid cells
pub const N_GRID_TOTAL: u32 = N_GRID * N_GRID;

pub const DX: f32 = 1.0 / (N_GRID as f32);
pub const INV_DX: f32 = N_GRID as f32;
pub const DT: f32 = 1e-4 / (QUALITY as f32);
pub const P_VOL: f32 = (DX * 0.5) * (DX * 0.5);
pub const P_RHO: f32 = 1.0;
pub const P_MASS: f32 = P_VOL * P_RHO;
pub const YOUNGS_MODULUS: f32 = 5e3;
pub const POISSON_RATIO: f32 = 0.2;

/// Lame parameter mu
pub const MU_0: f32 = YOUNGS_MODULUS / (2.0 * (1.0 + POISSON_RATIO));
/// Lame parameter lambda
pub const LAMBDA_0: f32 =
    YOUNGS_MODULUS * POISSON_RATIO / ((1.0 + POISSON_RATIO) * (1.0 - 2.0 * POISSON_RATIO));

/// Workgroup size for compute shaders
/// IMPORTANT: This must be kept in sync with the literal value in kernel/src/lib.rs
pub const WORKGROUP_SIZE: u32 = 64;
pub const GRID_WORKGROUP_SIZE: (u32, u32) = (8, 8);

#[inline]
pub const fn div_ceil_u32(n: u32, d: u32) -> u32 {
    // Precondition: d > 0
    n / d + ((n % d) != 0) as u32
}

pub fn num_workgroups_1d(num_elements: u32) -> [u32; 3] {
    [div_ceil_u32(num_elements, WORKGROUP_SIZE), 1, 1]
}

pub fn num_workgroups_2d(num_elts_x: u32, num_elts_y: u32) -> [u32; 3] {
    [
        div_ceil_u32(num_elts_x, GRID_WORKGROUP_SIZE.0),
        div_ceil_u32(num_elts_y, GRID_WORKGROUP_SIZE.1),
        1,
    ]
}

/// The constant used in the computation (index * 2 + COMPUTE_CONSTANT)
pub const COMPUTE_CONSTANT: u32 = 42;

/// Newtype wrapper for thread IDs to ensure type safety
#[derive(Copy, Clone, Debug)]
pub struct ThreadId(u32);

impl ThreadId {
    #[inline]
    #[cfg(not(any(target_arch = "spirv")))]
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    #[inline]
    #[cfg(any(target_arch = "spirv"))]
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    #[inline]
    pub fn as_u32(&self) -> u32 {
        self.0
    }

    #[inline]
    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }
}

/// Push constants shared between CPU and GPU
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PushConstants {
    pub num_elements: u32,
}

// Bitonic sort implementation
// A comparison-based sorting algorithm well-suited for parallel execution on GPUs

/// Newtype wrapper for bitonic sort stage
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
#[repr(transparent)]
pub struct Stage(pub u32);

impl Stage {
    #[inline]
    pub fn new(stage: u32) -> Self {
        Self(stage)
    }

    #[inline]
    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

/// Newtype wrapper for pass within a stage
#[derive(Copy, Clone, Debug, PartialEq, Eq, Pod, Zeroable)]
#[repr(transparent)]
pub struct Pass(pub u32);

impl Pass {
    #[inline]
    pub fn new(pass: u32) -> Self {
        Self(pass)
    }

    #[inline]
    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

/// Sort order
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(u32)]
pub enum SortOrder {
    Ascending = 0,
    Descending = 1,
}

impl Display for SortOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SortOrder::Ascending => write!(f, "ascending"),
            SortOrder::Descending => write!(f, "descending"),
        }
    }
}

impl TryFrom<u32> for SortOrder {
    type Error = &'static str;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(SortOrder::Ascending),
            1 => Ok(SortOrder::Descending),
            _ => Err("Invalid SortOrder value"),
        }
    }
}

impl From<SortOrder> for u32 {
    fn from(order: SortOrder) -> u32 {
        match order {
            SortOrder::Ascending => 0,
            SortOrder::Descending => 1,
        }
    }
}

/// Trait for types that can be sorted using bitonic sort
pub trait SortableKey: Copy + Pod + Zeroable + PartialOrd {
    /// Convert to sortable unsigned representation for GPU operations
    fn to_sortable_u32(&self) -> u32;

    /// Convert back from sortable representation
    fn from_sortable_u32(val: u32) -> Self;

    /// Compare two values for sorting
    fn should_swap(&self, other: &Self, order: SortOrder) -> bool {
        match order {
            SortOrder::Ascending => self > other,
            SortOrder::Descending => self < other,
        }
    }

    /// Get the maximum value for this type (used for padding)
    fn max_value() -> Self;

    /// Get the minimum value for this type (used for padding)
    fn min_value() -> Self;
}

// Implement SortableKey for u32
impl SortableKey for u32 {
    #[inline]
    fn to_sortable_u32(&self) -> u32 {
        *self
    }

    #[inline]
    fn from_sortable_u32(val: u32) -> Self {
        val
    }

    #[inline]
    fn max_value() -> Self {
        u32::MAX
    }

    #[inline]
    fn min_value() -> Self {
        u32::MIN
    }
}

// Implement SortableKey for i32
impl SortableKey for i32 {
    #[inline]
    fn to_sortable_u32(&self) -> u32 {
        // Flip sign bit to make negative numbers sort correctly
        (*self as u32) ^ (1 << 31)
    }

    #[inline]
    fn from_sortable_u32(val: u32) -> Self {
        (val ^ (1 << 31)) as i32
    }

    #[inline]
    fn max_value() -> Self {
        i32::MAX
    }

    #[inline]
    fn min_value() -> Self {
        i32::MIN
    }
}

// Implement SortableKey for f32
impl SortableKey for f32 {
    #[inline]
    fn to_sortable_u32(&self) -> u32 {
        let bits = self.to_bits();
        // If negative, flip all bits; if positive, flip just sign bit
        if bits & (1 << 31) != 0 {
            !bits
        } else {
            bits | (1 << 31)
        }
    }

    #[inline]
    fn from_sortable_u32(val: u32) -> Self {
        let bits = if val & (1 << 31) != 0 {
            val & !(1 << 31)
        } else {
            !val
        };
        f32::from_bits(bits)
    }

    #[inline]
    fn max_value() -> Self {
        f32::INFINITY
    }

    #[inline]
    fn min_value() -> Self {
        f32::NEG_INFINITY
    }
}

/// Parameters for GPU bitonic sorting
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct BitonicParams {
    pub num_elements: u32,
    pub stage: Stage,        // Current stage (for multi-dispatch approach)
    pub pass_of_stage: Pass, // Current pass within stage
    pub sort_order: u32,     // Sort order as u32 (0 = Ascending, 1 = Descending)
}

/// Direction for bitonic compare operations
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum CompareDirection {
    Up,   // Sort in ascending order
    Down, // Sort in descending order
}

impl CompareDirection {
    #[inline]
    pub fn from_bool(ascending: bool) -> Self {
        if ascending {
            CompareDirection::Up
        } else {
            CompareDirection::Down
        }
    }

    #[inline]
    pub fn is_ascending(&self) -> bool {
        matches!(self, CompareDirection::Up)
    }
}
