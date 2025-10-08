//! Runner implementations for different compute backends

pub mod vulkano;
pub use self::vulkano::VulkanoRunner;
