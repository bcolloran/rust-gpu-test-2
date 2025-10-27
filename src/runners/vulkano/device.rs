use crate::error::{ChimeraError, CrateResult};
use std::sync::Arc;

use vulkano::{
    device::{Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo},
    instance::{Instance, InstanceCreateInfo},
    VulkanLibrary,
};

pub fn compute_capable_device_and_queue(
) -> CrateResult<(Arc<Instance>, String, Arc<Device>, Arc<Queue>)> {
    // 1. Load the Vulkan library
    let library = VulkanLibrary::new()?;

    // 2. Create instance with surface extensions (required for graphics/swapchain)
    let mut instance_info = InstanceCreateInfo::default();
    // Enable surface extension and platform-specific window extensions
    instance_info.enabled_extensions = vulkano::instance::InstanceExtensions {
        khr_surface: true,
        khr_xlib_surface: true,    // For X11/Linux
        khr_xcb_surface: true,     // Alternative X11
        khr_wayland_surface: true, // For Wayland/Linux
        ..Default::default()
    };
    let instance = Instance::new(library, instance_info)?;

    // 3. Pick first physical device with a compute queue
    let physical = instance
        .enumerate_physical_devices()?
        .next()
        .ok_or_else(|| ChimeraError::NoVulkanDevice(0))?;

    let device_name = physical.properties().device_name.clone();

    // 4. Select a queue family that supports compute and graphics
    // This allows both compute and graphics operations on the same queue
    let (queue_family_index, _q_props) = physical
        .queue_family_properties()
        .iter()
        .enumerate()
        .find(|(_, q)| {
            q.queue_flags.contains(vulkano::device::QueueFlags::COMPUTE)
                && q.queue_flags
                    .contains(vulkano::device::QueueFlags::GRAPHICS)
        })
        .map(|(i, q)| (i as u32, q.clone()))
        .ok_or(ChimeraError::NoComputeQueue)?;

    // 5. Create logical device + queue
    // Enable storage buffer storage class extension (required by generated SPIR-V)
    // Also enable swapchain extension so graphics rendering can share this device
    // Reference: Vulkano book compute pipeline chapter
    let required_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        khr_swapchain: true, // Required for graphics rendering
        ext_shader_atomic_float: true,
        ..DeviceExtensions::empty()
    };

    // Enable required device features. Our SPIR-V (from rust-gpu targeting Vulkan 1.2)
    // uses the VulkanMemoryModel capability, which maps to the `vulkan_memory_model`
    // device feature. Without enabling this feature, Vulkano validation rejects the
    // shader module creation (previous runtime error root cause).
    let mut required_capabilities = DeviceFeatures::empty();
    required_capabilities.vulkan_memory_model = true;
    required_capabilities.vulkan_memory_model_device_scope = true;

    required_capabilities.shader_buffer_float32_atomics = true;
    required_capabilities.shader_buffer_float32_atomic_add = true;
    // required_features.shader_buffer_float32_atomic_min_max = true;

    required_capabilities.shader_buffer_float64_atomics = true;
    required_capabilities.shader_buffer_float64_atomic_add = true;
    // required_features.shader_buffer_float64_atomic_min_max = true;

    required_capabilities.shader_buffer_int64_atomics = true;
    required_capabilities.shader_int8 = true;

    // dbg!(physical.supported_features());
    // // Verify support before requesting so we can provide a clearer error.

    // required_features.check_requirements(&physical.supported_features());

    // if !physical.supported_features().contains(&required_features) {
    //     return Err(ChimeraError::Other(
    //         "Selected physical device does not support required feature".into(),
    //     ));
    // }

    let (device, mut queues) = Device::new(
        physical,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: required_extensions,
            enabled_features: required_capabilities,
            ..Default::default()
        },
    )?;

    let queue = queues
        .next()
        .ok_or_else(|| ChimeraError::Other("Failed to get compute queue".into()))?;

    Ok((instance, device_name, device, queue))
}
