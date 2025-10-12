//! Device and queue setup for graphics rendering
//!
//! This module handles finding and creating a Vulkan device that supports
//! both graphics operations and presentation to a window surface.

use crate::error::CrateResult;
use std::sync::Arc;
use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        DeviceExtensions, QueueFlags,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    swapchain::Surface,
    VulkanLibrary,
};

/// Find a suitable physical device and queue family for graphics + presentation
///
/// This function filters devices by:
/// 1. Support for the swapchain extension (required for presentation)
/// 2. A queue family that supports both GRAPHICS operations and surface presentation
/// 3. Preference for discrete GPUs over integrated/virtual/CPU devices
pub fn select_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
    device_extensions: &DeviceExtensions,
) -> CrateResult<(Arc<PhysicalDevice>, u32)> {
    instance
        .enumerate_physical_devices()?
        .filter(|p| p.supported_extensions().contains(device_extensions))
        .filter_map(|p| {
            // Find a queue family that supports both graphics and presentation
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, surface).unwrap_or(false)
                })
                .map(|q| (p, q as u32))
        })
        .min_by_key(|(p, _)| {
            // Prefer discrete GPUs, then integrated, then virtual, then CPU
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                _ => 4,
            }
        })
        .ok_or(crate::graphics::error::GraphicsError::NoSuitableDevice.into())
}

/// Create a Vulkan instance with extensions required for windowing
pub fn create_instance_for_windowing(
    library: Arc<VulkanLibrary>,
    required_extensions: vulkano::instance::InstanceExtensions,
) -> CrateResult<Arc<Instance>> {
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )?;
    Ok(instance)
}
