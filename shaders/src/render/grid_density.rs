use shared::P_MASS;
use spirv_std::{glam::Vec4, spirv};

/// Vertex shader for rendering the grid as a heatmap
///
/// This shader renders the grid using instanced quads. Each instance represents
/// one grid cell, and we generate a quad (2 triangles = 6 vertices) for each cell.
///
/// The shader:
/// 1. Uses instance_index to determine which grid cell this is
/// 2. Uses vertex_index (0-5) to determine which corner of the quad
/// 3. Reads the GridCell data (mass, velocity) from the storage buffer
/// 4. Positions the quad to cover the appropriate screen region
/// 5. Passes the mass value to the fragment shader for coloring
///
/// The grid is rendered behind the particles (drawn first in the command buffer).
#[spirv(vertex)]
pub fn grid_density_vs(
    #[spirv(vertex_index)] vert_idx: i32,
    #[spirv(instance_index)] inst_idx: i32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] grid: &[shared::grid::GridCell],
    #[spirv(push_constant)] push_constants: &shared::grid::GridPushConstants,
    #[spirv(position)] builtin_pos: &mut Vec4,
    out_mass: &mut f32,
) {
    let grid_width = push_constants.grid_width;
    let grid_height = push_constants.grid_height;

    // Calculate row and column from instance index
    let col = (inst_idx as u32) % grid_width;
    let row = (inst_idx as u32) / grid_width;

    // Read grid cell data
    let cell = grid[inst_idx as usize];
    let mass = cell.mass;

    // Calculate cell size in normalized coordinates [0, 1]
    let cell_width = 1.0 / (grid_width as f32);
    let cell_height = 1.0 / (grid_height as f32);

    // Calculate cell position in normalized coordinates [0, 1]
    let cell_x = (col as f32) * cell_width;
    let cell_y = (row as f32) * cell_height;

    // Generate quad vertices based on vertex_index
    // We use a triangle list with 6 vertices per quad:
    // 0,1,2 for first triangle, 3,4,5 for second triangle
    // Layout:
    //   0 --- 1/3
    //   |  \   |
    //   2/4 -- 5
    let local_vert_idx = vert_idx % 6;
    let (dx, dy) = match local_vert_idx {
        0 => (0.0, 0.0), // top-left
        1 => (1.0, 0.0), // top-right
        2 => (0.0, 1.0), // bottom-left
        3 => (1.0, 0.0), // top-right (second triangle)
        4 => (0.0, 1.0), // bottom-left (second triangle)
        5 => (1.0, 1.0), // bottom-right
        _ => (0.0, 0.0), // should never happen
    };

    // Calculate final position in [0, 1] space
    let pos_x = cell_x + dx * cell_width;
    let pos_y = cell_y + dy * cell_height;

    // Convert from [0, 1] to [-1, 1] clip space
    let clip_x = pos_x * 2.0 - 1.0;
    let clip_y = pos_y * 2.0 - 1.0;

    *builtin_pos = Vec4::new(clip_x, clip_y, 0.0, 1.0);
    *out_mass = mass;
}

/// Fragment shader for rendering the grid heatmap
///
/// This shader converts the mass value (assumed to be in [0, 1]) to a grayscale color.
/// Higher mass values appear brighter (whiter), lower values appear darker (blacker).
#[spirv(fragment)]
pub fn grid_density_fs(in_mass: f32, output: &mut Vec4) {
    let mass_clamped = if in_mass > 0.0 {
        // iterpolate between 0.0 at in_mass=0.0 to 1.0 at in_mass=P_MASS * MASS_MULTIPLIER, clamped to [0, 1]
        const MASS_MULTIPLIER: f32 = 20.0;
        let interpolated = in_mass / (P_MASS * MASS_MULTIPLIER);
        let clamped = interpolated.min(1.0);

        const COLOR_MIN: f32 = 0.1;
        const COLOR_MAX: f32 = 0.5;

        // If mass is positive, use minimum value of COLOR_MIN
        // saturate to COLOR_MAX when MASS_MULTIPLIER*P_MASS is reached
        // Clamp mass to [0, 1] range to be safe
        COLOR_MIN + (clamped * (COLOR_MAX - COLOR_MIN))
    } else {
        0.0
    };

    // Simple grayscale mapping: mass directly maps to brightness
    // You could add color mapping here for a more interesting heatmap
    *output = Vec4::new(mass_clamped, mass_clamped, mass_clamped, 1.0);
}
