//! Rust GPU Chimera Demo Library
//!
//! This library demonstrates running the same Rust code on:
//! - CPU (native Rust)
//! - Vulkan (via rust-gpu/SPIR-V)

#![feature(once_cell_try)]

pub mod error;
pub mod runners;
pub use runners::VulkanoRunner;

#[cfg(any(feature = "vulkano"))]
pub const OTHER_SHADERS_SPIRV: &[u8] = include_bytes!(env!("SHADERS_SPV_PATH"));
#[cfg(any(feature = "vulkano"))]
pub const SHADERS_ENTRY_ADDER: &str = env!("SHADERS_ENTRY_ADDER");
