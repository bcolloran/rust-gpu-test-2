use std::{collections::HashMap, sync::Arc};

use vulkano::shader::{EntryPoint, ShaderModule};

use crate::runners::vulkano::shader::shader_entry_point;

#[derive(Clone)]
pub struct EntryPointNameToBuffers {
    pub shaders: HashMap<String, BufNameToBinding>,
}

impl EntryPointNameToBuffers {
    pub fn from_lists(lists: Vec<(&str, Vec<(&str, u32)>)>) -> Self {
        let mut shaders = HashMap::new();
        for (entry_point_name, list) in lists {
            shaders.insert(
                entry_point_name.to_string(),
                BufNameToBinding::from_list(list),
            );
        }
        EntryPointNameToBuffers { shaders }
    }

    pub fn validate_against_global_buf_names(&self, buf_names_to_bindings: &BufNameToBinding) {
        for (entry_point_name, global_bufs) in &self.shaders {
            for buf_name in global_bufs.0.keys() {
                if !buf_names_to_bindings.0.contains_key(buf_name) {
                    panic!("Shader '{entry_point_name}' requires buffer '{buf_name}' which is not in the global buffer name to binding mapping.");
                }
            }
        }
        println!("\n✅ Shader buffer mappings validated against global buffer names.");
    }
}

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

pub struct EntryPointNameToBuffersAndEntryPoint {
    pub shaders: HashMap<String, (BufNameToBinding, EntryPoint)>,
}
impl EntryPointNameToBuffersAndEntryPoint {
    pub fn from_entry_point_names(
        shader_module: Arc<ShaderModule>,

        entry_point_name_to_buffers: &EntryPointNameToBuffers,
    ) -> Self {
        let mut shaders = HashMap::new();
        for (entry_point_name, buf_name_to_binding) in &entry_point_name_to_buffers.shaders {
            let entry_point = shader_entry_point(shader_module.clone(), entry_point_name)
                .unwrap_or_else(|e| {
                    panic!("Failed to get entry point '{entry_point_name}': {e}");
                });
            shaders.insert(
                entry_point_name.clone(),
                (buf_name_to_binding.clone(), entry_point),
            );
        }
        Self { shaders }
    }
}
