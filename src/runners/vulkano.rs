//! Vulkano runner implementation - Safe Vulkan abstraction via vulkano
//!
//! Mirrors the behaviour of the raw ash based runner but implemented using
//! the higher level `vulkano` crate. This keeps resource management safer
//! (RAII) and shortens the amount of boilerplate. The public behaviour is
//! intended to match `AshRunner` so it can be swapped transparently.

pub mod buffer;
pub mod descriptor_sets;
pub mod device;
pub mod pipeline;
pub mod shader;
pub mod shader_buffer_mapping;

use crate::{
    error::{ChimeraError, Result},
    runners::vulkano::{buffer::build_and_fill_buffer, device::compute_capable_device_and_queue},
    SortRunner, SHADERS_ENTRY_ADDER,
};
use shared::WORKGROUP_SIZE;
use std::sync::Arc;

// Vulkano imports (version 0.35 API)
use std::collections::BTreeMap;
use vulkano::{
    buffer::Subbuffer,
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryAutoCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator,
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType,
        },
        DescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue},
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        compute::{ComputePipeline, ComputePipelineCreateInfo},
        layout::{PipelineLayout, PipelineLayoutCreateInfo},
        Pipeline, PipelineBindPoint, PipelineShaderStageCreateInfo,
    },
    shader::ShaderStages,
    sync::{self, GpuFuture},
};

/// Vulkan-based runner for bitonic sort using vulkano safe abstractions
pub struct VulkanoRunner {
    device: Arc<Device>,
    queue: Arc<Queue>,
    adder_pipeline: Arc<ComputePipeline>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    device_name: String,
}

fn build_adder_pipeline(device: Arc<Device>) -> Result<Arc<ComputePipeline>> {
    let shader_module = shader::shader_module(device.clone())?;
    let entry_point = shader::shader_entry_point(shader_module, SHADERS_ENTRY_ADDER)?;

    // Descriptor set layout (binding 0: storage buffer)
    let mut binding0 = DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer);
    binding0.stages = ShaderStages::COMPUTE;
    binding0.descriptor_count = 1;

    let mut binding1 = DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer);
    binding1.stages = ShaderStages::COMPUTE;
    binding1.descriptor_count = 1;

    let mut bindings = BTreeMap::new();
    bindings.insert(0u32, binding0);
    bindings.insert(1u32, binding1);

    let descriptor_set_layout = DescriptorSetLayout::new(
        device.clone(),
        DescriptorSetLayoutCreateInfo {
            bindings,
            ..Default::default()
        },
    )?;

    // Pipeline layout + push constants
    let pipeline_layout = PipelineLayout::new(
        device.clone(),
        PipelineLayoutCreateInfo {
            set_layouts: vec![descriptor_set_layout],
            // push_constant_ranges: vec![PushConstantRange {
            //     stages: ShaderStages::COMPUTE,
            //     offset: 0,
            //     size: std::mem::size_of::<BitonicPushConstants>() as u32,
            // }],
            ..Default::default()
        },
    )?;

    // Build stage and compute pipeline
    let stage = PipelineShaderStageCreateInfo::new(entry_point.clone());
    let pipeline_info = ComputePipelineCreateInfo::stage_layout(stage, pipeline_layout.clone());
    let pipeline = ComputePipeline::new(device.clone(), None, pipeline_info)?;
    Ok(pipeline)
}

impl VulkanoRunner {
    /// Create a new Vulkano runner
    pub fn new() -> Result<Self> {
        let (device_name, device, queue) = compute_capable_device_and_queue()?;

        // 6. Memory allocator
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let adder_pipeline = build_adder_pipeline(device.clone())?;

        Ok(Self {
            device,
            queue,
            adder_pipeline,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            device_name,
        })
    }

    /// Small helper to build a descriptor set for (a, rhs)
    fn make_adder_set(
        &self,
        layout: std::sync::Arc<vulkano::descriptor_set::layout::DescriptorSetLayout>,
        a: Subbuffer<[u32]>,
        rhs: Subbuffer<[u32]>,
    ) -> Result<std::sync::Arc<DescriptorSet>> {
        let writes = [
            WriteDescriptorSet::buffer(0, a),
            WriteDescriptorSet::buffer(1, rhs),
        ];
        let set = DescriptorSet::new(self.descriptor_set_allocator.clone(), layout, writes, [])?;
        Ok(set)
    }

    fn bind_descriptor_sets_and_dispatch(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        set: Arc<DescriptorSet>,
        num_wg: u32,
    ) -> Result<()> {
        builder.bind_descriptor_sets(
            PipelineBindPoint::Compute,
            self.adder_pipeline.layout().clone(),
            0,
            set,
        )?;
        unsafe {
            builder.dispatch([num_wg, 1, 1])?;
        }
        Ok(())
    }

    fn run_adder_pass(&self, a: &mut [u32], b: &[u32], c: &[u32], d: &[u32]) -> Result<()> {
        assert_eq!(a.len(), b.len());
        // Allocate a CPU visible buffer, copy input, run compute, read back
        let len = a.len();
        let num_workgroups = (len as u32).div_ceil(WORKGROUP_SIZE);

        let alloc = self.memory_allocator.clone();
        let buffer_a = build_and_fill_buffer(alloc.clone(), a)?;
        let buffer_b = build_and_fill_buffer(alloc.clone(), b)?;
        let buffer_c = build_and_fill_buffer(alloc.clone(), c)?;
        let buffer_d = build_and_fill_buffer(alloc.clone(), d)?;

        // Create descriptor set (binding 0: storage buffer)
        let layout = self
            .adder_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .cloned()
            .ok_or_else(|| {
                ChimeraError::Other("Pipeline missing descriptor set layout 0".into())
            })?;

        let set_ab = self.make_adder_set(layout.clone(), buffer_a.clone(), buffer_b.clone())?;
        let set_ac = self.make_adder_set(layout.clone(), buffer_a.clone(), buffer_c.clone())?;
        let set_ad = self.make_adder_set(layout.clone(), buffer_a.clone(), buffer_d.clone())?;

        // Build command buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        println!("`Ok(builder)`");
        builder.bind_pipeline_compute(self.adder_pipeline.clone())?;

        self.bind_descriptor_sets_and_dispatch(&mut builder, set_ab, num_workgroups)?;
        self.bind_descriptor_sets_and_dispatch(&mut builder, set_ac, num_workgroups)?;
        self.bind_descriptor_sets_and_dispatch(&mut builder, set_ad, num_workgroups)?;

        let command_buffer = builder.build()?;

        // Execute + wait
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?;
        future.wait(None)?;

        // Read back results (buffer is host visible)
        let content = buffer_a.read()?;
        a.copy_from_slice(&content[..len]);

        Ok(())
    }
}

//
//
//
//
//
//
//
//
//

impl SortRunner for VulkanoRunner {
    fn backend_info(
        &self,
    ) -> (
        &'static str,
        Option<&'static str>,
        Option<String>,
        Option<String>,
    ) {
        (
            "vulkano",
            Some("Vulkan"),
            Some(self.device_name.clone()),
            None,
        )
    }

    // fn execute_kernel_pass(&self, data: &mut [u32], params: BitonicParams) -> Result<()> {
    //     self.run_pass(data, params)
    // }

    fn execute_adder_kernel_pass(
        &self,
        a: &mut [u32],
        b: &[u32],
        c: &[u32],
        d: &[u32],
    ) -> Result<()> {
        self.run_adder_pass(a, b, c, d)
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
