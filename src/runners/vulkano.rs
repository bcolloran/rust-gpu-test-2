pub mod buffer;
pub mod descriptor_sets;
pub mod device;
pub mod dispatch;
pub mod pipeline;
pub mod shader;
pub mod shader_buffer_mapping;

use crate::{
    error::CrateResult,
    runners::vulkano::{
        buffer::{build_and_fill_buffer, BufNameToBufferAny},
        device::compute_capable_device_and_queue,
        shader_buffer_mapping::{
            ComputePassInvocationInfo, ShaderPipelineInfosWithComputePipelines,
        },
    },
};
use glam::Vec2;
use shared::WORKGROUP_SIZE;
use std::{collections::HashMap, sync::Arc};

use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{Device, Queue},
    memory::allocator::StandardMemoryAllocator,
    sync::{self, GpuFuture},
};

/// Vulkan-based runner for bitonic sort using vulkano safe abstractions
pub struct VulkanoRunner {
    device: Arc<Device>,
    queue: Arc<Queue>,
    compute_pipelines: ShaderPipelineInfosWithComputePipelines,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,

    device_name: String,
}

impl VulkanoRunner {
    /// Create a new Vulkano runner
    pub fn new(entry_point_names_to_buffers: ComputePassInvocationInfo) -> CrateResult<Self> {
        let (device_name, device, queue) = compute_capable_device_and_queue()?;
        println!("Using device: {}", device_name);

        let shader_module = shader::shader_module(device.clone())?;
        println!("Shader module created");

        //   allocators
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        println!("Allocators created");

        let shader_bufs_and_entries =
            entry_point_names_to_buffers.with_entry_points(shader_module.clone());

        println!("Shader entry points created");

        let descriptor_set_layouts =
            shader_bufs_and_entries.with_descriptor_sets(device.clone())?;
        println!("Descriptor set layouts created");

        let compute_pipelines = descriptor_set_layouts.with_pipelines(device.clone())?;

        println!("VulkanoRunner::new ok");
        Ok(Self {
            device,
            queue,
            compute_pipelines,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            device_name,
        })
    }

    pub fn execute_adder_kernel_pass(
        &self,
        a: &mut [u32],
        b: &[u32],
        c: &[u32],
        d: &[u32],
        x: &mut [Vec2],
        v: &[Vec2],
    ) -> CrateResult<()> {
        assert_eq!(a.len(), b.len());
        // Allocate a CPU visible buffer, copy input, run compute, read back
        let len = a.len();
        let num_workgroups = (len as u32).div_ceil(WORKGROUP_SIZE);

        let alloc = self.memory_allocator.clone();
        let buffer_a = build_and_fill_buffer(alloc.clone(), a)?;
        let buffer_b = build_and_fill_buffer(alloc.clone(), b)?;
        let buffer_c = build_and_fill_buffer(alloc.clone(), c)?;
        let buffer_d = build_and_fill_buffer(alloc.clone(), d)?;
        let buffer_x = build_and_fill_buffer(alloc.clone(), x)?;
        let buffer_v = build_and_fill_buffer(alloc.clone(), v)?;

        let buffers = BufNameToBufferAny(HashMap::from([
            ("a".to_string(), buffer_a.into()),
            ("b".to_string(), buffer_b.into()),
            ("c".to_string(), buffer_c.into()),
            ("d".to_string(), buffer_d.into()),
            ("x".to_string(), buffer_x.into()),
            ("v".to_string(), buffer_v.into()),
        ]));

        let compute_pipelines_with_desc_sets = self
            .compute_pipelines
            .with_descriptor_sets(self.descriptor_set_allocator.clone(), &buffers)?;

        // Build command buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        compute_pipelines_with_desc_sets
            .bind_and_dispatch_all(&mut builder, num_workgroups)
            .inspect_err(|e| println!("Error during bind_and_dispatch_all: {e}"))?;

        println!("  Dispatching compute on device '{}'", self.device_name);

        let command_buffer = builder.build()?;

        // Execute + wait
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?;
        future.wait(None)?;

        // Read back results (buffer is host visible)
        let content = &buffers.0["a"].read_u32()[..len];

        a.copy_from_slice(&content);
        x.copy_from_slice(&buffers.0["x"].read_vec2()[..len]);

        Ok(())
    }
}

// #[cfg(test)]
// mod tests {
//     use super::VulkanoRunner;
//     use crate::{verify_sorted, SortRunner};
//     use shared::SortOrder;

//     #[test]
//     fn test_bitonic_u32() {
//         let runner = VulkanoRunner::new().unwrap();
//         let mut data = vec![42u32, 7, 999, 0, 13, 256, 128, 511];

//         runner.sort(&mut data, SortOrder::Ascending).unwrap();
//         assert!(verify_sorted(&data, SortOrder::Ascending));
//         assert_eq!(data, vec![0, 7, 13, 42, 128, 256, 511, 999]);
//     }

//     #[test]
//     fn test_bitonic_i32() {
//         let runner = VulkanoRunner::new().unwrap();
//         let mut data = vec![-42i32, 7, -999, 0, 13, -256, 128, -1];

//         runner.sort(&mut data, SortOrder::Ascending).unwrap();
//         assert!(verify_sorted(&data, SortOrder::Ascending));
//         assert_eq!(data, vec![-999, -256, -42, -1, 0, 7, 13, 128]);
//     }

//     #[test]
//     fn test_bitonic_f32() {
//         let runner = VulkanoRunner::new().unwrap();
//         let mut data = vec![3.14f32, -2.71, 0.0, -0.0, 1.41, -99.9, 42.0];

//         runner.sort(&mut data, SortOrder::Ascending).unwrap();
//         assert!(verify_sorted(&data, SortOrder::Ascending));
//     }

//     #[test]
//     fn test_bitonic_u32_descending() {
//         let runner = VulkanoRunner::new().unwrap();
//         let mut data = vec![42u32, 7, 999, 0, 13, 256, 128, 511];

//         runner.sort(&mut data, SortOrder::Descending).unwrap();
//         assert!(verify_sorted(&data, SortOrder::Descending));
//         assert_eq!(data, vec![999, 511, 256, 128, 42, 13, 7, 0]);
//     }

//     #[test]
//     fn test_bitonic_i32_descending() {
//         let runner = VulkanoRunner::new().unwrap();
//         let mut data = vec![-42i32, 7, -999, 0, 13, -256, 128, -1];

//         runner.sort(&mut data, SortOrder::Descending).unwrap();
//         assert!(verify_sorted(&data, SortOrder::Descending));
//         assert_eq!(data, vec![128, 13, 7, 0, -1, -42, -256, -999]);
//     }

//     #[test]
//     fn test_bitonic_f32_descending() {
//         let runner = VulkanoRunner::new().unwrap();
//         let mut data = vec![3.14f32, -2.71, 0.0, -0.0, 1.41, -99.9, 42.0];

//         runner.sort(&mut data, SortOrder::Descending).unwrap();
//         assert!(verify_sorted(&data, SortOrder::Descending));
//     }
// }
