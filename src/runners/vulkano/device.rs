use crate::error::{ChimeraError, CrateResult};
use std::sync::Arc;

use vulkano::{
    device::{Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo},
    instance::{Instance, InstanceCreateInfo},
    VulkanLibrary,
};

pub fn compute_capable_device_and_queue() -> CrateResult<(String, Arc<Device>, Arc<Queue>)> {
    // 1. Load the Vulkan library
    let library = VulkanLibrary::new()?;

    // 2. Create instance
    let instance = Instance::new(library, InstanceCreateInfo::default())?;

    // 3. Pick first physical device with a compute queue
    let physical = instance
        .enumerate_physical_devices()?
        .next()
        .ok_or_else(|| ChimeraError::NoVulkanDevice(0))?;

    let device_name = physical.properties().device_name.clone();

    // 4. Select a queue family that supports compute
    let (queue_family_index, _q_props) = physical
        .queue_family_properties()
        .iter()
        .enumerate()
        .find(|(_, q)| q.queue_flags.contains(vulkano::device::QueueFlags::COMPUTE))
        .map(|(i, q)| (i as u32, q.clone()))
        .ok_or(ChimeraError::NoComputeQueue)?;

    // 5. Create logical device + queue
    // Enable storage buffer storage class extension (required by generated SPIR-V)
    // Reference: Vulkano book compute pipeline chapter
    let required_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::empty()
    };

    // Enable required device features. Our SPIR-V (from rust-gpu targeting Vulkan 1.2)
    // uses the VulkanMemoryModel capability, which maps to the `vulkan_memory_model`
    // device feature. Without enabling this feature, Vulkano validation rejects the
    // shader module creation (previous runtime error root cause).
    let mut required_features = DeviceFeatures::empty();
    required_features.vulkan_memory_model = true;

    // Verify support before requesting so we can provide a clearer error.
    if !physical.supported_features().contains(&required_features) {
        return Err(ChimeraError::Other(
            "Selected physical device does not support required feature: vulkan_memory_model"
                .into(),
        ));
    }

    let (device, mut queues) = Device::new(
        physical,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: required_extensions,
            enabled_features: required_features,
            ..Default::default()
        },
    )?;

    let queue = queues
        .next()
        .ok_or_else(|| ChimeraError::Other("Failed to get compute queue".into()))?;

    Ok((device_name, device, queue))
}
