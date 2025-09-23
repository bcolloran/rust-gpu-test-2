//! Error types for the library

use thiserror::Error;

/// Error types for the Rust GPU Chimera demo
#[derive(Error, Debug)]
pub enum ChimeraError {
    #[error("No suitable GPU adapter found")]
    NoAdapter,

    #[error("No suitable Vulkan device found among {0} devices")]
    NoVulkanDevice(usize),

    #[error("Buffer size overflow: {0} elements × {1} bytes per element")]
    BufferSizeOverflow(usize, usize),

    #[error("Mapped memory size ({mapped}) is smaller than expected ({expected})")]
    InsufficientMappedMemory { mapped: u64, expected: u64 },

    #[error("Failed to find kernel module: {0}")]
    KernelNotFound(String),

    #[error("Failed to find compute queue family")]
    NoComputeQueue,

    #[cfg(feature = "wgpu")]
    #[error("wgpu error: {0}")]
    Wgpu(#[from] wgpu::Error),

    #[cfg(feature = "wgpu")]
    #[error("wgpu request device error: {0}")]
    WgpuRequestDevice(#[from] wgpu::RequestDeviceError),

    #[cfg(feature = "ash")]
    #[error("Vulkan error: {0}")]
    Vulkan(String),

    #[cfg(feature = "vulkano")]
    #[error("vulkano error: {0}")]
    Vulkano(String),

    #[cfg(feature = "vulkano")]
    #[error("vulkano LoadingError: {0}")]
    VulkanoLoadingError(#[from] vulkano::LoadingError),

    #[cfg(feature = "vulkano")]
    #[error("vulkano VulkanError: {0}")]
    VulkanoVulkanError(#[from] vulkano::VulkanError),

    #[cfg(feature = "vulkano")]
    #[error("vulkano ValidationError: {0}")]
    VulkanoValidationError(#[from] vulkano::ValidationError),

    // #[cfg(feature = "vulkano")]
    // #[error("vulkano VulkanError: {0}")]
    // VulkanoVulkanError(#[from] vulkano::VulkanError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Other error: {0}")]
    Other(String),
}

impl From<Box<dyn std::error::Error>> for ChimeraError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        ChimeraError::Other(err.to_string())
    }
}

#[cfg(feature = "ash")]
impl From<ash::vk::Result> for ChimeraError {
    fn from(err: ash::vk::Result) -> Self {
        ChimeraError::Vulkan(format!("Vulkan error: {err:?}"))
    }
}

#[cfg(feature = "vulkano")]
impl From<vulkano::Validated<vulkano::VulkanError>> for ChimeraError {
    fn from(err: vulkano::Validated<vulkano::VulkanError>) -> Self {
        ChimeraError::Vulkano(err.to_string())
    }
}

// Additional Vulkano-specific error conversions used by the runner
#[cfg(feature = "vulkano")]
impl From<vulkano::shader::spirv::SpirvBytesNotMultipleOf4> for ChimeraError {
    fn from(err: vulkano::shader::spirv::SpirvBytesNotMultipleOf4) -> Self {
        ChimeraError::Vulkano(err.to_string())
    }
}

#[cfg(feature = "vulkano")]
impl From<Box<vulkano::ValidationError>> for ChimeraError {
    fn from(err: Box<vulkano::ValidationError>) -> Self {
        ChimeraError::Vulkano(err.to_string())
    }
}

/// Convenience type alias for Results with [`ChimeraError`]
pub type Result<T> = std::result::Result<T, ChimeraError>;
