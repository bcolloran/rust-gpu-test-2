use spirv_std::{arch::atomic_f_add, glam::Vec2};

pub unsafe fn atomic_f_add_vec2<const SCOPE: u32, const SEMANTICS: u32>(dst: &mut Vec2, val: Vec2) {
    atomic_f_add::<_, SCOPE, SEMANTICS>(&mut dst.x, val.x);
    atomic_f_add::<_, SCOPE, SEMANTICS>(&mut dst.y, val.y);
}
