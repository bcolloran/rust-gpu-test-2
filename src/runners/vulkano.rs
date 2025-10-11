pub mod buffer;
pub mod descriptor_sets;
pub mod device;
pub mod dispatch;
pub mod pipeline;
pub mod shader;
pub mod shader_buffer_mapping;

use crate::{
    error::{ChimeraError, Result},
    runners::vulkano::{
        buffer::build_and_fill_buffer,
        descriptor_sets::build_abstract_descriptor_set_layout,
        device::compute_capable_device_and_queue,
        dispatch::bind_and_dispatch,
        pipeline::build_pipeline,
        shader_buffer_mapping::{
            BufNameToBinding, EntryPointNameToBuffers, EntryPointNameToBuffersAndEntryPoint,
        },
    },
    SortRunner,
};
use glam::Vec2;
use shared::WORKGROUP_SIZE;
use std::{collections::HashMap, sync::Arc};

use vulkano::{
    buffer::Subbuffer,
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, layout::DescriptorSetLayout, DescriptorSet,
        WriteDescriptorSet,
    },
    device::{Device, Queue},
    memory::allocator::StandardMemoryAllocator,
    pipeline::compute::ComputePipeline,
    sync::{self, GpuFuture},
};

/// Vulkan-based runner for bitonic sort using vulkano safe abstractions
pub struct VulkanoRunner {
    device: Arc<Device>,
    queue: Arc<Queue>,
    compute_pipelines: HashMap<String, Arc<ComputePipeline>>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    // descriptor set layout shared by all pipelines
    shared_descriptor_set_layout: Arc<DescriptorSetLayout>,
    device_name: String,
}

impl VulkanoRunner {
    /// Create a new Vulkano runner
    pub fn new(
        global_buf_to_binding: BufNameToBinding,
        entry_point_names_to_buffers: EntryPointNameToBuffers,
    ) -> Result<Self> {
        let (device_name, device, queue) = compute_capable_device_and_queue()?;

        let shader_module = shader::shader_module(device.clone())?;

        let shader_bufs_and_entries = EntryPointNameToBuffersAndEntryPoint::from_entry_point_names(
            shader_module,
            &entry_point_names_to_buffers,
        );

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

        let descriptor_set_layout =
            build_abstract_descriptor_set_layout(device.clone(), global_buf_to_binding)?;

        let compute_pipelines: HashMap<String, Arc<ComputePipeline>> = entry_point_names_to_buffers
            .shaders
            .iter()
            .filter_map(|(name, _)| {
                let entry = shader_bufs_and_entries
                    .shaders
                    .get(name)
                    .ok_or_else(|| {
                        ChimeraError::Other(format!("No entry point found for '{name}'"))
                    })
                    .ok()?
                    .1
                    .clone();
                let pipeline =
                    build_pipeline(device.clone(), descriptor_set_layout.clone(), entry).ok()?;
                Some((name.clone(), pipeline))
            })
            .collect::<_>();

        Ok(Self {
            device,
            queue,
            compute_pipelines,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            shared_descriptor_set_layout: descriptor_set_layout,
            device_name,
        })
    }

    /// Small helper to build a descriptor set for (a, rhs)
    fn make_adder_set(
        &self,
        layout: Arc<DescriptorSetLayout>,
        a: Subbuffer<[u32]>,
        rhs: Subbuffer<[u32]>,
    ) -> Result<Arc<DescriptorSet>> {
        let writes = [
            WriteDescriptorSet::buffer(0, a),
            WriteDescriptorSet::buffer(1, rhs),
        ];
        let set = DescriptorSet::new(self.descriptor_set_allocator.clone(), layout, writes, [])?;
        Ok(set)
    }

    fn run_adder_pass(
        &self,
        a: &mut [u32],
        b: &[u32],
        c: &[u32],
        d: &[u32],
        x: &[Vec2],
        v: &[Vec2],
    ) -> Result<()> {
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

        // Create descriptor set (binding 0: storage buffer)
        let layout = self.shared_descriptor_set_layout.clone();

        let set_ab = self.make_adder_set(layout.clone(), buffer_a.clone(), buffer_b.clone())?;
        let set_ac = self.make_adder_set(layout.clone(), buffer_a.clone(), buffer_c.clone())?;
        let set_ad = self.make_adder_set(layout.clone(), buffer_a.clone(), buffer_d.clone())?;

        // Build command buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        builder.bind_pipeline_compute(self.compute_pipelines["adder"].clone())?;

        bind_and_dispatch(
            &mut builder,
            self.compute_pipelines["adder"].clone(),
            set_ab,
            num_workgroups,
        )?;

        bind_and_dispatch(
            &mut builder,
            self.compute_pipelines["adder"].clone(),
            set_ac,
            num_workgroups,
        )?;

        bind_and_dispatch(
            &mut builder,
            self.compute_pipelines["adder"].clone(),
            set_ad,
            num_workgroups,
        )?;

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

    fn execute_adder_kernel_pass(
        &self,
        a: &mut [u32],
        b: &[u32],
        c: &[u32],
        d: &[u32],
        x: &mut [Vec2],
        v: &[Vec2],
    ) -> Result<()> {
        self.run_adder_pass(a, b, c, d, x, v)
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
