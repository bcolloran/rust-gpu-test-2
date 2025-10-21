use crate::error::CrateResult;
use std::{collections::HashMap, ops::Index, sync::Arc};

use glam::Vec2;
use shared::grid::GridCell;
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    descriptor_set::WriteDescriptorSet,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
};

#[allow(non_camel_case_types)]
pub enum BufferKindsEnum {
    u32(Subbuffer<[u32]>),
    f32(Subbuffer<[f32]>),
    Vec2(Subbuffer<[Vec2]>),
    GridCell(Subbuffer<[GridCell]>),
}

impl From<Subbuffer<[u32]>> for BufferKindsEnum {
    fn from(buf: Subbuffer<[u32]>) -> Self {
        BufferKindsEnum::u32(buf)
    }
}
impl From<Subbuffer<[f32]>> for BufferKindsEnum {
    fn from(buf: Subbuffer<[f32]>) -> Self {
        BufferKindsEnum::f32(buf)
    }
}
impl From<Subbuffer<[Vec2]>> for BufferKindsEnum {
    fn from(buf: Subbuffer<[Vec2]>) -> Self {
        BufferKindsEnum::Vec2(buf)
    }
}
impl From<Subbuffer<[GridCell]>> for BufferKindsEnum {
    fn from(buf: Subbuffer<[GridCell]>) -> Self {
        BufferKindsEnum::GridCell(buf)
    }
}

impl TryInto<Subbuffer<[u32]>> for BufferKindsEnum {
    type Error = String;

    fn try_into(self) -> std::result::Result<Subbuffer<[u32]>, Self::Error> {
        match self {
            BufferKindsEnum::u32(buf) => Ok(buf),
            _ => Err("BufferAny is not of type u32".to_string()),
        }
    }
}
impl TryInto<Subbuffer<[f32]>> for BufferKindsEnum {
    type Error = String;

    fn try_into(self) -> std::result::Result<Subbuffer<[f32]>, Self::Error> {
        match self {
            BufferKindsEnum::f32(buf) => Ok(buf),
            _ => Err("BufferAny is not of type f32".to_string()),
        }
    }
}
impl TryInto<Subbuffer<[Vec2]>> for BufferKindsEnum {
    type Error = String;

    fn try_into(self) -> std::result::Result<Subbuffer<[Vec2]>, Self::Error> {
        match self {
            BufferKindsEnum::Vec2(buf) => Ok(buf),
            _ => Err("BufferAny is not of type Vec2".to_string()),
        }
    }
}
impl TryInto<Subbuffer<[GridCell]>> for BufferKindsEnum {
    type Error = String;

    fn try_into(self) -> std::result::Result<Subbuffer<[GridCell]>, Self::Error> {
        match self {
            BufferKindsEnum::GridCell(buf) => Ok(buf),
            _ => Err("BufferAny is not of type GridCell".to_string()),
        }
    }
}

impl BufferKindsEnum {
    pub fn write_descriptor_set_for_binding(&self, binding: u32) -> WriteDescriptorSet {
        match self {
            BufferKindsEnum::u32(buf) => WriteDescriptorSet::buffer(binding, buf.clone()),
            BufferKindsEnum::f32(buf) => WriteDescriptorSet::buffer(binding, buf.clone()),
            BufferKindsEnum::Vec2(buf) => WriteDescriptorSet::buffer(binding, buf.clone()),
            BufferKindsEnum::GridCell(buf) => WriteDescriptorSet::buffer(binding, buf.clone()),
        }
    }

    pub fn read_u32(&self) -> Vec<u32> {
        match self {
            BufferKindsEnum::u32(buf) => {
                let content = buf.read().expect("failed to read buffer");
                content.to_vec()
            }
            _ => panic!("BufferAny is not of type u32"),
        }
    }
    pub fn read_f32(&self) -> Vec<f32> {
        match self {
            BufferKindsEnum::f32(buf) => {
                let content = buf.read().expect("failed to read buffer");
                content.to_vec()
            }
            _ => panic!("BufferAny is not of type f32"),
        }
    }
    pub fn read_vec2(&self) -> Vec<Vec2> {
        match self {
            BufferKindsEnum::Vec2(buf) => {
                let content = buf.read().expect("failed to read buffer");
                content.to_vec()
            }
            _ => panic!("BufferAny is not of type Vec2"),
        }
    }
    pub fn read_grid_cell(&self) -> Vec<GridCell> {
        match self {
            BufferKindsEnum::GridCell(buf) => {
                let content = buf.read().expect("failed to read buffer");
                content.to_vec()
            }
            _ => panic!("BufferAny is not of type GridCell"),
        }
    }
}

pub struct BufNameToBufferAny(pub HashMap<String, BufferKindsEnum>);

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
