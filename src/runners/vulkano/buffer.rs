use crate::error::Result;
use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
};

pub fn build_and_fill_buffer<T: BufferContents + Copy>(
    memory_allocator: Arc<StandardMemoryAllocator>,
    data: &[T],
) -> Result<Subbuffer<[T]>> {
    let usage = BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST;

    let buffer: Subbuffer<[T]> = Buffer::from_iter(
        memory_allocator,
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
