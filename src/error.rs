//! Error types for the library

use thiserror::Error;
#[cfg(feature = "vulkano")]
use vulkano::{
    buffer::AllocateBufferError, command_buffer::CommandBufferExecError, sync::HostAccessError,
    Validated,
};

/// Error types for the Rust GPU Chimera demo
#[derive(Error, Debug)]
pub enum ChimeraError {
    #[error("Environment variable error: {0}")]
    VarError(#[from] std::env::VarError),

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

    #[error("vulkano error: {0}")]
    Vulkano(String),

    #[error("vulkano LoadingError: {0}")]
    VulkanoLoadingError(#[from] vulkano::LoadingError),

    #[error("vulkano VulkanError: {0}")]
    VulkanoVulkanError(#[from] vulkano::VulkanError),

    #[error("vulkano ValidationError: {0}")]
    VulkanoValidationError(#[from] vulkano::ValidationError),

    #[error("vulkano SpirvBytesNotMultipleOf4: {0}")]
    VulkanoSpirvBytesNotMultipleOf4(#[from] vulkano::shader::spirv::SpirvBytesNotMultipleOf4),

    #[error("vulkano CommandBufferExecError: {0}")]
    VulkanoCommandBufferExecError(#[from] CommandBufferExecError),

    #[error("vulkano HostAccessError: {0}")]
    VulkanoHostAccessError(#[from] HostAccessError),

    #[error("vulkano ValidatedAllocateBufferError: {0}")]
    VulkanoValidatedAllocateBufferError(#[from] Validated<AllocateBufferError>),

    #[error("vulkano ValidatedAllocateBufferError: {0}")]
    VulkanoBoxedValidationError(#[from] Box<vulkano::ValidationError>),

    #[error("vulkano VulkanoValidatedValidationError: {0}")]
    VulkanoValidatedValidationError(#[from] Validated<vulkano::ValidationError>),

    #[error("vulkano VulkanoValidatedVulkanError: {0}")]
    VulkanoValidatedVulkanError(#[from] Validated<vulkano::VulkanError>),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Graphics error: {0}")]
    Graphics(#[from] crate::graphics::error::GraphicsError),

    #[error("Other error: {0}")]
    Other(String),
}

impl From<Box<dyn std::error::Error>> for ChimeraError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        ChimeraError::Other(err.to_string())
    }
}

/// Convenience type alias for Results with [`ChimeraError`]
pub type CrateResult<T> = std::result::Result<T, ChimeraError>;
