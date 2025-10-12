use thiserror::Error;

/// Error types specific to the graphics rendering subsystem
#[derive(Error, Debug)]
pub enum GraphicsError {
    #[error("No suitable physical device found for graphics rendering")]
    NoSuitableDevice,

    #[error("Vertex shader entry point '{0}' not found in shader module")]
    VertexShaderEntryPointNotFound(String),

    #[error("Fragment shader entry point '{0}' not found in shader module")]
    FragmentShaderEntryPointNotFound(String),

    #[error("Failed to create render pass subpass")]
    SubpassCreationFailed,

    #[error("No queue available from device")]
    NoQueueAvailable,

    #[error("No surface formats available for the given surface")]
    NoSurfaceFormatsAvailable,

    #[error("No composite alpha modes available for the given surface")]
    NoCompositeAlphaModesAvailable,

    #[error("Queue family {0} does not support presentation to the surface")]
    QueueDoesNotSupportPresentation(u32),

    #[error("No descriptor set layout found at index {0} in pipeline")]
    NoDescriptorSetLayout(usize),
}
