//! Runner implementations for different compute backends

pub mod cpu;

#[cfg(feature = "wgpu")]
pub mod wgpu;

#[cfg(feature = "ash")]
pub mod ash;

#[cfg(feature = "vulkano")]
pub mod vulkano;

// Re-export runners at module level for convenience
pub use cpu::CpuRunner;

#[cfg(feature = "wgpu")]
pub use self::wgpu::WgpuRunner;

#[cfg(feature = "ash")]
pub use self::ash::AshRunner;

#[cfg(feature = "vulkano")]
pub use self::vulkano::VulkanoRunner;
