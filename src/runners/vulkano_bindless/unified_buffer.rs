/// Unified buffer management for bindless rendering
///
/// In the bindless approach, instead of creating separate GPU buffers for each
/// logical buffer (a, b, c, d, x, v, etc.), we pack them all into two large
/// "unified" buffers:
/// 1. One unified buffer for all u32 data (a, b, c, d, etc.)
/// 2. One unified buffer for all Vec2 data (x, v, etc.)
///
/// Each logical buffer gets an offset within its unified buffer, and we track
/// these offsets so we can tell the shader where to find each logical buffer.
use crate::error::CrateResult;
use glam::Vec2;
use std::{collections::HashMap, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
};

/// Tracks where each logical buffer is located within the unified buffers
pub struct UnifiedBufferTracker {
    /// Offsets (in elements) for u32 logical buffers
    pub u32_offsets: HashMap<String, u32>,
    /// Offsets (in elements) for Vec2 logical buffers  
    pub vec2_offsets: HashMap<String, u32>,
    /// The actual unified u32 buffer on the GPU
    pub unified_u32_buffer: Subbuffer<[u32]>,
    /// The actual unified Vec2 buffer on the GPU
    pub unified_vec2_buffer: Subbuffer<[Vec2]>,
    /// Size of each logical buffer (they're all the same size in our case)
    pub logical_buffer_size: usize,
}

impl UnifiedBufferTracker {
    /// Create unified buffers from the provided data
    ///
    /// This function:
    /// 1. Determines which buffers contain u32 vs Vec2 data
    /// 2. Packs all u32 buffers into one large buffer
    /// 3. Packs all Vec2 buffers into one large buffer
    /// 4. Records the offset of each logical buffer
    ///
    /// # Arguments
    /// * `memory_allocator` - Vulkan memory allocator
    /// * `buffer_data` - Map of buffer names to their data (as slices)
    pub fn new(
        memory_allocator: Arc<StandardMemoryAllocator>,
        a: &[u32],
        b: &[u32],
        c: &[u32],
        d: &[u32],
        x: &[Vec2],
        v: &[Vec2],
    ) -> CrateResult<Self> {
        // All buffers must be the same size
        let size = a.len();
        assert_eq!(b.len(), size);
        assert_eq!(c.len(), size);
        assert_eq!(d.len(), size);
        assert_eq!(x.len(), size);
        assert_eq!(v.len(), size);

        // Build unified u32 buffer by concatenating all u32 buffers
        // Layout: [a_data][b_data][c_data][d_data]
        let mut unified_u32_data = Vec::with_capacity(size * 4);
        let mut u32_offsets = HashMap::new();

        // Add buffer 'a' at offset 0
        u32_offsets.insert("a".to_string(), 0u32);
        unified_u32_data.extend_from_slice(a);

        // Add buffer 'b' at offset size
        u32_offsets.insert("b".to_string(), size as u32);
        unified_u32_data.extend_from_slice(b);

        // Add buffer 'c' at offset 2*size
        u32_offsets.insert("c".to_string(), (size * 2) as u32);
        unified_u32_data.extend_from_slice(c);

        // Add buffer 'd' at offset 3*size
        u32_offsets.insert("d".to_string(), (size * 3) as u32);
        unified_u32_data.extend_from_slice(d);

        // Build unified Vec2 buffer by concatenating all Vec2 buffers
        // Layout: [x_data][v_data]
        let mut unified_vec2_data = Vec::with_capacity(size * 2);
        let mut vec2_offsets = HashMap::new();

        // Add buffer 'x' at offset 0
        vec2_offsets.insert("x".to_string(), 0u32);
        unified_vec2_data.extend_from_slice(x);

        // Add buffer 'v' at offset size
        vec2_offsets.insert("v".to_string(), size as u32);
        unified_vec2_data.extend_from_slice(v);

        // Create the actual GPU buffers
        let unified_u32_buffer = create_buffer(&memory_allocator, &unified_u32_data)?;
        let unified_vec2_buffer = create_buffer(&memory_allocator, &unified_vec2_data)?;

        Ok(Self {
            u32_offsets,
            vec2_offsets,
            unified_u32_buffer,
            unified_vec2_buffer,
            logical_buffer_size: size,
        })
    }

    /// Get the offset for a u32 buffer
    pub fn get_u32_offset(&self, buffer_name: &str) -> u32 {
        *self
            .u32_offsets
            .get(buffer_name)
            .unwrap_or_else(|| panic!("u32 buffer '{}' not found in unified buffer", buffer_name))
    }

    /// Get the offset for a Vec2 buffer
    pub fn get_vec2_offset(&self, buffer_name: &str) -> u32 {
        *self
            .vec2_offsets
            .get(buffer_name)
            .unwrap_or_else(|| panic!("Vec2 buffer '{}' not found in unified buffer", buffer_name))
    }

    /// Read back a u32 logical buffer from the unified buffer
    pub fn read_u32_buffer(&self, buffer_name: &str) -> Vec<u32> {
        let offset = self.get_u32_offset(buffer_name) as usize;
        let size = self.logical_buffer_size;

        let content = self
            .unified_u32_buffer
            .read()
            .expect("Failed to read unified u32 buffer");

        content[offset..offset + size].to_vec()
    }

    /// Read back a Vec2 logical buffer from the unified buffer
    pub fn read_vec2_buffer(&self, buffer_name: &str) -> Vec<Vec2> {
        let offset = self.get_vec2_offset(buffer_name) as usize;
        let size = self.logical_buffer_size;

        let content = self
            .unified_vec2_buffer
            .read()
            .expect("Failed to read unified Vec2 buffer");

        content[offset..offset + size].to_vec()
    }
}

/// Helper function to create a GPU buffer from data
fn create_buffer<T: BufferContents + Copy>(
    memory_allocator: &Arc<StandardMemoryAllocator>,
    data: &[T],
) -> CrateResult<Subbuffer<[T]>> {
    let usage = BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST;

    let buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        data.iter().copied(),
    )?;

    Ok(buffer)
}
