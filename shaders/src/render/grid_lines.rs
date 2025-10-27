use spirv_std::{glam::Vec4, spirv};

/// Vertex shader for rendering grid lines
///
/// This shader renders the grid lines using the LineList topology.
/// Each pair of vertices defines one line segment. We generate lines for:
/// - Vertical lines: one line for each column boundary (grid_width + 1 lines)
/// - Horizontal lines: one line for each row boundary (grid_height + 1 lines)
///
/// The shader:
/// 1. Uses vertex_index to determine which line and which endpoint (start/end)
/// 2. Uses push constants to get grid dimensions
/// 3. Generates line endpoints in normalized [0, 1] coordinates
/// 4. Converts to clip space [-1, 1]
///
/// Total vertices needed:
/// - Vertical lines: (grid_width + 1) * 2 vertices
/// - Horizontal lines: (grid_height + 1) * 2 vertices
#[spirv(vertex)]
pub fn grid_lines_vs(
    #[spirv(vertex_index)] vert_idx: i32,
    #[spirv(push_constant)] push_constants: &shared::grid::GridPushConstants,
    #[spirv(position)] builtin_pos: &mut Vec4,
) {
    let grid_width = push_constants.grid_width;
    let grid_height = push_constants.grid_height;

    // Total number of vertical line vertices
    let num_vertical_verts = (grid_width + 1) * 2;

    let (pos_x, pos_y) = if (vert_idx as u32) < num_vertical_verts {
        // Vertical lines
        let line_idx = (vert_idx as u32) / 2; // Which vertical line (0 to grid_width)
        let is_start = (vert_idx % 2) == 0; // Start or end of line

        // X position is at the column boundary
        let x = (line_idx as f32) / (grid_width as f32);

        // Y position: start at top (0.0), end at bottom (1.0)
        let y = if is_start { 0.0 } else { 1.0 };

        (x, y)
    } else {
        // Horizontal lines
        let horiz_vert_idx = (vert_idx as u32) - num_vertical_verts;
        let line_idx = horiz_vert_idx / 2; // Which horizontal line (0 to grid_height)
        let is_start = (vert_idx % 2) == 0; // Start or end of line

        // Y position is at the row boundary
        let y = (line_idx as f32) / (grid_height as f32);

        // X position: start at left (0.0), end at right (1.0)
        let x = if is_start { 0.0 } else { 1.0 };

        (x, y)
    };

    // Convert from [0, 1] to [-1, 1] clip space
    let clip_x = pos_x * 2.0 - 1.0;
    let clip_y = pos_y * 2.0 - 1.0;

    *builtin_pos = Vec4::new(clip_x, clip_y, 0.0, 1.0);
}

/// Fragment shader for rendering grid lines
///
/// This outputs a light red color for all grid line pixels.
#[spirv(fragment)]
pub fn grid_lines_fs(output: &mut Vec4) {
    // Light red color (RGB: ~1.0, 0.5, 0.5)
    *output = Vec4::new(0.5, 0.1, 0.1, 0.5);
}
