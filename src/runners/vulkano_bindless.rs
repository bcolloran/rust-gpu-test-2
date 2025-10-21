pub mod buffer;
pub mod descriptor_sets;
pub mod device;
pub mod dispatch;
pub mod pipeline;
pub mod shader;
pub mod shader_buffer_mapping;
pub mod unified_buffer;

use self::shader_buffer_mapping::BindlessComputePass;
use self::unified_buffer::UnifiedBufferTracker;
use crate::{
    error::CrateResult,
    runners::vulkano::{
        device::compute_capable_device_and_queue, shader_buffer_mapping::ComputePassInvocationInfo,
    },
};

use glam::Vec2;
use shared::{GRID_WORKGROUP_SIZE, WORKGROUP_SIZE};

use std::sync::Arc;

use vulkano::{
    buffer::Subbuffer,
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{Device, Queue},
    instance::Instance,
    memory::allocator::StandardMemoryAllocator,
    sync::{self, GpuFuture},
};

/// Vulkan-based runner for compute shaders using a bindless approach
///
/// Unlike the traditional "bindfull" approach where each logical buffer gets its own
/// descriptor binding, this runner uses a bindless approach where:
/// - All u32 data is packed into one unified buffer
/// - All Vec2 data is packed into another unified buffer  
/// - Shaders use push constants to know where each logical buffer starts
/// - Only 2 descriptor bindings are needed total (vs 6+ in the bindfull approach)
pub struct VulkanoBindlessRunner {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    /// Information about the compute pass (shader entry points, buffer mappings, etc.)
    compute_pass_info: ComputePassInvocationInfo,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    device_name: String,
}

impl VulkanoBindlessRunner {
    /// Create a new bindless Vulkano runner
    ///
    /// Unlike the traditional runner which creates descriptor sets and pipelines
    /// during initialization, the bindless runner defers most of that work until
    /// the actual compute call. This is because:
    /// 1. We don't know buffer sizes/data until run_compute_and_get_buffer is called
    /// 2. The bindless approach creates unified buffers from the actual data
    /// 3. Pipelines and descriptor sets are simpler (only 2 bindings) so recreating isn't expensive
    pub fn new(compute_pass_info: ComputePassInvocationInfo) -> CrateResult<Self> {
        let (instance, device_name, device, queue) = compute_capable_device_and_queue()?;
        println!("Using device: {}", device_name);

        // Create allocators
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        println!("VulkanoBindlessRunner::new ok");
        Ok(Self {
            instance,
            device,
            queue,
            compute_pass_info,
            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,
            device_name,
        })
    }

    /// Run compute shaders using the bindless approach and return the x buffer for graphics rendering
    ///
    /// This method implements the bindless paradigm:
    /// 1. Creates unified buffers (one for all u32 data, one for all Vec2 data)
    /// 2. Builds compute pipelines with only 2 descriptor bindings
    /// 3. Uses push constants to tell shaders where each logical buffer is located
    /// 4. Executes the compute pass
    /// 5. Reads back results from the unified buffers
    ///
    /// The key difference from the bindfull approach is that we only bind 2 descriptor
    /// sets total (one for u32 buffer, one for Vec2 buffer), and use push constants
    /// to specify offsets for each dispatch.
    pub fn run_compute_and_get_buffer(
        &self,
        a: &mut [u32],
        b: &[u32],
        c: &[u32],
        d: &[u32],
        x: &mut [Vec2],
        v: &[Vec2],
    ) -> CrateResult<(Subbuffer<[Vec2]>, usize)> {
        assert_eq!(a.len(), b.len());
        let len = a.len();
        let num_workgroups = (len as u32).div_ceil(WORKGROUP_SIZE);

        // BINDLESS STEP 1: Create unified buffers
        // Instead of 6 separate buffers, we create 2 unified buffers
        let unified_buffers =
            UnifiedBufferTracker::new(self.memory_allocator.clone(), a, b, c, d, x, v)?;

        // BINDLESS STEP 2: Create the compute pass with bindless pipelines
        // This will create pipelines that expect unified buffers and push constants
        let compute_pass = BindlessComputePass::new(
            self.device.clone(),
            &self.compute_pass_info,
            self.descriptor_set_allocator.clone(),
            &unified_buffers,
        )?;

        // BINDLESS STEP 3: Build command buffer and dispatch shaders
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        // Execute all the compute shader dispatches
        // Each dispatch will use push constants to specify buffer offsets
        compute_pass
            .dispatch_all(&mut builder, num_workgroups, &unified_buffers)
            .inspect_err(|e| println!("Error during dispatch_all: {e}"))?;

        // println!("  Dispatching compute on device '{}'", self.device_name);

        let command_buffer = builder.build()?;

        // BINDLESS STEP 4: Execute on GPU
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?;
        future.wait(None)?;

        // BINDLESS STEP 5: Read back results from unified buffers
        // Extract the logical buffers from the unified buffers
        let a_result = unified_buffers.read_u32_buffer("a");
        a.copy_from_slice(&a_result);

        let x_result = unified_buffers.read_vec2_buffer("x");
        x.copy_from_slice(&x_result);

        // For graphics rendering, we need to return a buffer containing just 'x'.
        // Since x is part of the unified buffer, we need to extract it.
        // For simplicity, we'll create a new buffer with just x data.
        // (In a real application, you might want to keep using the unified buffer with offsets)
        let x_standalone = crate::runners::vulkano::buffer::build_and_fill_buffer(
            self.memory_allocator.clone(),
            &x_result,
        )?;

        Ok((x_standalone, len))
    }

    /// Get the Vulkan instance (needed for creating windows/surfaces)
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }

    /// Get the device (useful for creating graphics resources on the same device)
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Get the queue (useful for graphics rendering on the same queue)
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    /// Get the memory allocator
    pub fn memory_allocator(&self) -> &Arc<StandardMemoryAllocator> {
        &self.memory_allocator
    }
}
