use crate::error::CrateResult;
use std::{collections::HashMap, ops::Index, sync::Arc};

use glam::Vec2;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    descriptor_set::WriteDescriptorSet,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
};

#[allow(non_camel_case_types)]
pub enum BufferAny {
    u32(Subbuffer<[u32]>),
    f32(Subbuffer<[f32]>),
    Vec2(Subbuffer<[Vec2]>),
}

impl From<Subbuffer<[u32]>> for BufferAny {
    fn from(buf: Subbuffer<[u32]>) -> Self {
        BufferAny::u32(buf)
    }
}
impl From<Subbuffer<[f32]>> for BufferAny {
    fn from(buf: Subbuffer<[f32]>) -> Self {
        BufferAny::f32(buf)
    }
}
impl From<Subbuffer<[Vec2]>> for BufferAny {
    fn from(buf: Subbuffer<[Vec2]>) -> Self {
        BufferAny::Vec2(buf)
    }
}

impl TryInto<Subbuffer<[u32]>> for BufferAny {
    type Error = String;

    fn try_into(self) -> std::result::Result<Subbuffer<[u32]>, Self::Error> {
        match self {
            BufferAny::u32(buf) => Ok(buf),
            _ => Err("BufferAny is not of type u32".to_string()),
        }
    }
}
impl TryInto<Subbuffer<[f32]>> for BufferAny {
    type Error = String;

    fn try_into(self) -> std::result::Result<Subbuffer<[f32]>, Self::Error> {
        match self {
            BufferAny::f32(buf) => Ok(buf),
            _ => Err("BufferAny is not of type f32".to_string()),
        }
    }
}
impl TryInto<Subbuffer<[Vec2]>> for BufferAny {
    type Error = String;

    fn try_into(self) -> std::result::Result<Subbuffer<[Vec2]>, Self::Error> {
        match self {
            BufferAny::Vec2(buf) => Ok(buf),
            _ => Err("BufferAny is not of type Vec2".to_string()),
        }
    }
}

impl BufferAny {
    pub fn write_descriptor_set_for_binding(&self, binding: u32) -> WriteDescriptorSet {
        match self {
            BufferAny::u32(buf) => WriteDescriptorSet::buffer(binding, buf.clone()),
            BufferAny::f32(buf) => WriteDescriptorSet::buffer(binding, buf.clone()),
            BufferAny::Vec2(buf) => WriteDescriptorSet::buffer(binding, buf.clone()),
        }
    }

    pub fn read_u32(&self) -> Vec<u32> {
        match self {
            BufferAny::u32(buf) => {
                let content = buf.read().expect("failed to read buffer");
                content.to_vec()
            }
            _ => panic!("BufferAny is not of type u32"),
        }
    }
    pub fn read_f32(&self) -> Vec<f32> {
        match self {
            BufferAny::f32(buf) => {
                let content = buf.read().expect("failed to read buffer");
                content.to_vec()
            }
            _ => panic!("BufferAny is not of type f32"),
        }
    }
    pub fn read_vec2(&self) -> Vec<Vec2> {
        match self {
            BufferAny::Vec2(buf) => {
                let content = buf.read().expect("failed to read buffer");
                content.to_vec()
            }
            _ => panic!("BufferAny is not of type Vec2"),
        }
    }
}

pub struct BufNameToBufferAny(pub HashMap<String, BufferAny>);

#[derive(Clone)]
pub struct BufNameToBinding(pub HashMap<String, u32>);
impl BufNameToBinding {
    pub fn from_list(list: Vec<(&str, u32)>) -> Self {
        let mut map = HashMap::new();
        for (buf_name, binding) in list {
            if map.insert(buf_name.to_string(), binding).is_some() {
                panic!("Duplicate buffer name in shader buffer mapping: {buf_name}");
            }
        }
        BufNameToBinding(map)
    }
}

impl Index<&str> for BufNameToBinding {
    type Output = u32;

    fn index(&self, index: &str) -> &Self::Output {
        &self.0[index]
    }
}

pub fn build_and_fill_buffer<T: BufferContents + Copy>(
    memory_allocator: Arc<StandardMemoryAllocator>,
    data: &[T],
) -> CrateResult<Subbuffer<[T]>> {
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

pub fn build_and_fill_buffer_and_get_write_descriptor_set<T: BufferContents + Copy>(
    memory_allocator: Arc<StandardMemoryAllocator>,
    data: &[T],
    buffer_index: u32,
) -> CrateResult<(Subbuffer<[T]>, WriteDescriptorSet)> {
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

    let write_descriptor_set = WriteDescriptorSet::buffer(buffer_index, buffer.clone());

    Ok((buffer, write_descriptor_set))
}
