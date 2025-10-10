use std::collections::HashMap;

pub struct ShaderBufferMapping {
    pub shaders: HashMap<String, GlobalBufNameToBinding>,
}
impl ShaderBufferMapping {
    pub fn from_lists(lists: Vec<(&str, Vec<(&str, u32)>)>) -> Self {
        let mut shaders = HashMap::new();
        for (shader_name, list) in lists {
            shaders.insert(
                shader_name.to_string(),
                GlobalBufNameToBinding::from_list(list),
            );
        }
        ShaderBufferMapping { shaders }
    }
}

pub struct GlobalBufNameToBinding(pub HashMap<String, u32>);
impl GlobalBufNameToBinding {
    pub fn from_list(list: Vec<(&str, u32)>) -> Self {
        let mut map = HashMap::new();
        for (buf_name, binding) in list {
            if map.insert(buf_name.to_string(), binding).is_some() {
                panic!("Duplicate buffer name in shader buffer mapping: {buf_name}");
            }
        }
        GlobalBufNameToBinding(map)
    }
}
