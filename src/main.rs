//! Demo showing compute shaders generating particle positions and graphics rendering them

use std::sync::Arc;

use anyhow::Result;
use bytemuck::Zeroable;
use glam::Vec2;
use rand::random;
use rust_gpu_chimera_demo::{
    graphics::GraphicsRenderer,
    runners::{
        vulkano::{
            buffer_specs::{buf_spec, DescriptorSetByName, IntoDescriptorSetByName},
            shader_pipeline_builder::{invoc_spec, kernel},
            typed_subbuffer_by_name::TypedSubbufferByName,
        },
        vulkano_compute_chain::VulkanoComputeChain,
    },
    *,
};
use shared::{
    grid::GridCell, num_workgroups_1d, num_workgroups_2d, DX, MATERIAL_GROUP_SIZE, N_GRID,
    N_PARTICLES,
};
use vulkano::{shader::ShaderModule, swapchain::Surface};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

// Application state
struct App<BS>
where
    BS: IntoDescriptorSetByName<Out: 'static + DescriptorSetByName + TypedSubbufferByName>,
{
    window: Option<Arc<Window>>,
    renderer: Option<GraphicsRenderer>,
    compute_chain: Option<VulkanoComputeChain<BS>>,

    frame_count: usize,
}

impl<BS> App<BS>
where
    BS: IntoDescriptorSetByName<Out: 'static + DescriptorSetByName + TypedSubbufferByName>,
{
    fn new(runner: VulkanoComputeChain<BS>, frame_count: usize) -> Self {
        Self {
            window: None,
            renderer: None,
            compute_chain: Some(runner),
            frame_count,
        }
    }
}

impl<BS> ApplicationHandler for App<BS>
where
    BS: IntoDescriptorSetByName<Out: 'static + DescriptorSetByName + TypedSubbufferByName>,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return; // Already initialized
        }

        println!("Creating window...");
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Rust GPU - Compute + Graphics Demo")
                        .with_inner_size(winit::dpi::LogicalSize::new(2000, 2000)),
                )
                .unwrap(),
        );

        // Use the instance from the compute runner (shared instance/device)
        let instance = self.compute_chain.as_ref().unwrap().instance().clone();
        let surface = Surface::from_window(instance, window.clone()).unwrap();

        // Load shader module for graphics
        use vulkano::shader::spirv::bytes_to_words;
        let spirv_words = bytes_to_words(OTHER_SHADERS_SPIRV).unwrap();
        let shader_module = unsafe {
            ShaderModule::new(
                self.compute_chain.as_ref().unwrap().device().clone(),
                vulkano::shader::ShaderModuleCreateInfo::new(&spirv_words),
            )
            .unwrap()
        };

        // Create graphics renderer using the compute device and queue
        let compute_chain = self.compute_chain.as_ref().unwrap();
        let mut renderer = GraphicsRenderer::from_device(
            compute_chain.device().clone(),
            compute_chain.queue().clone(),
            surface.clone(),
            shader_module.clone(),
            shader_module.clone(),
        )
        .unwrap();

        // Run compute once initially
        compute_chain.execute().unwrap();

        // Set up particle positions
        let buffer_x = compute_chain.typed_subbuffer_by_name::<Vec2>("x").unwrap();
        let num_particles = buffer_x.len() as usize;
        renderer
            .set_position_buffer(buffer_x, num_particles)
            .unwrap();

        // Set up grid buffer for heatmap rendering
        let grid_buffer = compute_chain
            .typed_subbuffer_by_name::<GridCell>("grid")
            .unwrap();
        // Grid is N x N where N is the number of particles (since grid is n*n elements)
        let grid_size = (num_particles as f64).sqrt() as u32;
        renderer
            .set_grid_buffer(grid_buffer, grid_size, grid_size)
            .unwrap();

        self.window = Some(window);
        self.renderer = Some(renderer);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let window = match self.window.as_ref() {
            Some(w) => w,
            None => return,
        };

        let renderer = match self.renderer.as_mut() {
            Some(r) => r,
            None => return,
        };

        let compute_chain = self.compute_chain.as_ref().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                println!(
                    "\nWindow closed. Total frames rendered: {}",
                    self.frame_count
                );
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                println!("Window resized to {}x{}", size.width, size.height);
                renderer.set_window_resized();
                if let Err(e) = renderer.handle_resize([size.width, size.height]) {
                    eprintln!("Resize error: {}", e);
                }
            }
            WindowEvent::RedrawRequested => {
                // Run compute shader to update particle positions and grid
                compute_chain.execute().unwrap();

                let buffer_x = compute_chain.typed_subbuffer_by_name::<Vec2>("x").unwrap();
                let num_particles = buffer_x.len() as usize;
                renderer
                    .set_position_buffer(buffer_x, num_particles)
                    .unwrap();

                // Update grid buffer (it's regenerated each frame by fill_grid_random)
                let grid_buffer = compute_chain
                    .typed_subbuffer_by_name::<GridCell>("grid")
                    .unwrap();

                renderer
                    .set_grid_buffer(grid_buffer, N_GRID, N_GRID)
                    .unwrap();

                // Render the frame
                if let Err(e) = renderer.render_frame() {
                    eprintln!("Render error: {}", e);
                }

                self.frame_count += 1;

                // Print progress every 60 frames
                if self.frame_count % 60 == 0 {
                    let x_buf = self
                        .compute_chain
                        .as_ref()
                        .unwrap()
                        .typed_subbuffer_by_name::<Vec2>("x")
                        .unwrap();
                    let x_read = x_buf.read().unwrap();
                    let x_slice = &x_read[0..3];

                    let grid_buf = self
                        .compute_chain
                        .as_ref()
                        .unwrap()
                        .typed_subbuffer_by_name::<GridCell>("grid")
                        .unwrap();
                    let g_read = grid_buf.read().unwrap();
                    let g_slice = &g_read[0..3];

                    println!("Frame {}", self.frame_count);
                    println!("Particles (x) :\n   {:?}", x_slice);
                    println!("GridCell buffer contents:\n   {:?}", g_slice)
                }

                // Request next frame
                window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    println!("=== Rust GPU Compute + Graphics Demo ===\n");

    let mut a = vec![1u32; N_PARTICLES as usize];
    let mut b = (0..N_PARTICLES as u32).collect::<Vec<u32>>();
    let mut c = vec![30u32; N_PARTICLES as usize];
    let mut d = (0..N_PARTICLES as u32).map(|x| x * x).collect::<Vec<u32>>();

    // Create particle positions and velocities
    // Positions will be moved around by compute shaders
    // let mut x = (0..n as u32)
    //     .map(|i| {
    //         let angle = (i as f32) * std::f32::consts::PI * 2.0 / n as f32;
    //         let radius = 0.3 + 0.2 * (i as f32 / n as f32);
    //         Vec2::new(0.5 + radius * angle.cos(), 0.5 + radius * angle.sin())
    //     })
    //     .collect::<Vec<Vec2>>();

    // let mut x = (0..N_PARTICLES as u32)
    //     .map(|i| {
    //         let group_offset = (i / MATERIAL_GROUP_SIZE) as f32;

    //         let px = random::<f32>() * 0.2 + 0.3 + 0.1 * group_offset;
    //         let py = random::<f32>() * 0.2 + 0.05 + 0.3 * group_offset;

    //         Vec2::new(px, py)
    //     })
    //     .collect::<Vec<Vec2>>();

    // Velocities - make particles spiral outward
    // let mut v = (0..N_PARTICLES as u32)
    //     .map(|i| {
    //         let angle = (i as f32) * std::f32::consts::PI * 2.0 / N_PARTICLES as f32;
    //         Vec2::new(0.00001 * angle.cos(), 0.00001 * angle.sin())
    //     })
    // .collect::<Vec<_>>();

    let mut x = (0..N_PARTICLES as u32)
        .map(|i| {
            let group_offset = (i / MATERIAL_GROUP_SIZE) as f32;
            let px = ((5 * i) as f32 + 0.5) * DX;
            Vec2::new(px, (0.5 + (5.0 * group_offset)) * DX)
        })
        .collect::<Vec<Vec2>>();

    let mut v = (0..N_PARTICLES as u32)
        .map(|_i| Vec2::new(0.000001, 0.0))
        .collect::<Vec<_>>();

    let mut grid = (0..(N_GRID * N_GRID))
        .map(|_| GridCell::zeroed())
        .collect::<Vec<_>>();

    let buf_specs = (
        buf_spec("a", 0, &mut a),
        buf_spec("b", 1, &mut b),
        buf_spec("c", 2, &mut c),
        buf_spec("d", 3, &mut d),
        // particles
        buf_spec("x", 2, &mut x),
        buf_spec("v", 3, &mut v),
        // grid
        buf_spec("grid", 4, &mut grid),
    );

    let wg_1d = num_workgroups_1d(N_PARTICLES as u32);
    let wg_2d = num_workgroups_2d(N_PARTICLES as u32, N_PARTICLES as u32);

    // Setup compute shader configuration
    let adder_kernel = kernel("adder", vec![0, 1], wg_1d);
    let step_particles_kernel = kernel("step_particles", vec![2, 3], wg_1d);
    let wrap_particles_kernel = kernel("wrap_particles", vec![2], wg_1d);

    let fill_grid_random_kernel = kernel("fill_grid_random", vec![4], wg_2d);
    let clear_grid_kernel = kernel("clear_grid", vec![4], wg_2d);
    let p2g_simple_test_kernel = kernel("p2g_simple_test", vec![2, 4], wg_1d);
    let p2g_kernel = kernel("p2g::p2g", vec![2, 3, 4], wg_1d);

    let invocation_chain = vec![
        invoc_spec("adder_ab", vec!["a", "b"], adder_kernel.clone()),
        invoc_spec(
            "step_particles_0",
            vec!["x", "v"],
            step_particles_kernel.clone(),
        ),
        invoc_spec(
            "step_particles_1",
            vec!["x", "v"],
            step_particles_kernel.clone(),
        ),
        invoc_spec(
            "step_particles_2",
            vec!["x", "v"],
            step_particles_kernel.clone(),
        ),
        invoc_spec(
            "step_particles_3",
            vec!["x", "v"],
            step_particles_kernel.clone(),
        ),
        invoc_spec("wrap_particles", vec!["x"], wrap_particles_kernel.clone()),
        // invoc_spec(
        //     "fill_grid_random",
        //     vec!["grid"],
        //     fill_grid_random_kernel.clone(),
        // ),
        invoc_spec("clear_grid", vec!["grid"], clear_grid_kernel.clone()),
        // invoc_spec("p2g_simple_test", vec!["x", "grid"], p2g_simple_test_kernel.clone()),
        invoc_spec("p2g", vec!["x", "v", "grid"], p2g_kernel.clone()),
    ];

    // Create compute runner
    println!("Initializing Vulkan compute...");
    let compute_chain = VulkanoComputeChain::new(&buf_specs, invocation_chain)?;
    println!("Compute runner initialized!");

    // Create application state
    let mut app = App::new(compute_chain, 0);

    // Create event loop and run
    let event_loop = EventLoop::new()?;
    event_loop.run_app(&mut app)?;

    Ok(())
}
