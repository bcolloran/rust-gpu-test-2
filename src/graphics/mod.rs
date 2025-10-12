//! Graphics module for rendering Vec2 points to the screen
//!
//! This module provides a simple graphics pipeline that:
//! - Creates a window and Vulkan swapchain for presenting images
//! - Uses the vertex and fragment shaders compiled from rust-gpu (shaders/src/lib.rs)
//! - Renders Vec2 points from a buffer as individual pixels on screen
//!
//! The pipeline is designed to be relatively independent from the compute pipeline,
//! though it shares the same Vulkan device and can access the same buffers.

pub mod device;
pub mod error;
pub mod pipeline;
pub mod renderer;

pub use renderer::GraphicsRenderer;
