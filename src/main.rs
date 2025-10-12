//! Demo showing compute shaders generating particle positions and graphics rendering them

use std::sync::Arc;

use anyhow::Result;
use glam::Vec2;
use rust_gpu_chimera_demo::{
    graphics::GraphicsRenderer,
    runners::vulkano::shader_buffer_mapping::ComputePassInvocationInfo,
    *,
};
use vulkano::{
    shader::ShaderModule,
    swapchain::Surface,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

// Application state
struct App {
    window: Option<Arc<Window>>,
    renderer: Option<GraphicsRenderer>,
    runner: Option<VulkanoRunner>,
    
    // Particle data
    a: Vec<u32>,
    b: Vec<u32>,
    c: Vec<u32>,
    d: Vec<u32>,
    x: Vec<Vec2>,
    v: Vec<Vec2>,
    
    frame_count: usize,
}

impl App {
    fn new(runner: VulkanoRunner, a: Vec<u32>, b: Vec<u32>, c: Vec<u32>, d: Vec<u32>, x: Vec<Vec2>, v: Vec<Vec2>) -> Self {
        Self {
            window: None,
            renderer: None,
            runner: Some(runner),
            a,
            b,
            c,
            d,
            x,
            v,
            frame_count: 0,
        }
    }
}

impl ApplicationHandler for App {
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
                        .with_inner_size(winit::dpi::LogicalSize::new(800, 600))
                )
                .unwrap()
        );

        // Use the instance from the compute runner (shared instance/device)
        println!("Creating surface from shared instance...");
        let instance = self.runner.as_ref().unwrap().instance().clone();
        let surface = Surface::from_window(instance, window.clone()).unwrap();

        // Load shader module for graphics
        println!("Loading graphics shaders...");
        use vulkano::shader::spirv::bytes_to_words;
        let spirv_words = bytes_to_words(OTHER_SHADERS_SPIRV).unwrap();
        let shader_module = unsafe {
            ShaderModule::new(
                self.runner.as_ref().unwrap().device().clone(),
                vulkano::shader::ShaderModuleCreateInfo::new(&spirv_words)
            ).unwrap()
        };

        // Create graphics renderer using the compute device and queue
        println!("Creating graphics renderer...");
        let runner = self.runner.as_ref().unwrap();
        let mut renderer = GraphicsRenderer::from_device(
            runner.device().clone(),
            runner.queue().clone(),
            surface.clone(),
            shader_module.clone(),
            shader_module.clone(),
        ).unwrap();

        println!("\n✓ Setup complete! Starting render loop...\n");
        println!("Controls:");
        println!("  - Close window to exit");
        println!("  - Particles will move and wrap around based on compute shaders");
        println!();

        // Run compute once initially
        println!("Running initial compute pass...");
        let runner = self.runner.as_ref().unwrap();
        let (buffer_x, num_particles) = runner.run_compute_and_get_buffer(
            &mut self.a, &self.b, &self.c, &self.d, &mut self.x, &self.v
        ).unwrap();
        
        println!("Particles after compute: {:?}", &self.x[..5.min(self.x.len())]);
        renderer.set_position_buffer(buffer_x, num_particles).unwrap();

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

        let runner = self.runner.as_ref().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                println!("\nWindow closed. Total frames rendered: {}", self.frame_count);
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
                // Run compute shader to update particle positions
                match runner.run_compute_and_get_buffer(
                    &mut self.a, &self.b, &self.c, &self.d, &mut self.x, &self.v
                ) {
                    Ok((buffer_x, num_particles)) => {
                        if let Err(e) = renderer.set_position_buffer(buffer_x, num_particles) {
                            eprintln!("Error setting position buffer: {}", e);
                        }
                    }
                    Err(e) => {
                        eprintln!("Compute error: {}", e);
                    }
                }

                // Render the frame
                if let Err(e) = renderer.render_frame() {
                    eprintln!("Render error: {}", e);
                }

                self.frame_count += 1;
                
                // Print progress every 60 frames
                if self.frame_count % 60 == 0 {
                    println!("Frame {}: particle 0 at {:?}", self.frame_count, self.x[0]);
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

    // Setup compute shader configuration
    let adder_kernel = ("adder", vec![0, 1]);
    let step_particles_kernel = ("step_particles", vec![2, 3]);
    let wrap_particles_kernel = ("wrap_particles", vec![2]);

    let shader_buffers = ComputePassInvocationInfo::from_lists(vec![
        ("adder_ab", vec!["a", "b"], adder_kernel.clone()),
        ("adder_ac", vec!["a", "c"], adder_kernel.clone()),
        ("adder_ad", vec!["a", "d"], adder_kernel.clone()),
        (
            "step_particles_0",
            vec!["x", "v"],
            step_particles_kernel.clone(),
        ),
        (
            "step_particles_1",
            vec!["x", "v"],
            step_particles_kernel.clone(),
        ),
        (
            "step_particles_2",
            vec!["x", "v"],
            step_particles_kernel.clone(),
        ),
        (
            "step_particles_3",
            vec!["x", "v"],
            step_particles_kernel.clone(),
        ),
        ("wrap_particles", vec!["x"], wrap_particles_kernel.clone()),
    ]);

    // Initialize particle data
    let n = 256;

    let a = vec![1u32; n];
    let b = (0..n as u32).collect::<Vec<u32>>();
    let c = vec![30u32; n];
    let d = (0..n as u32).map(|x| x * x).collect::<Vec<u32>>();

    // Create particle positions and velocities
    // Positions will be moved around by compute shaders
    let x = (0..n as u32)
        .map(|i| {
            let angle = (i as f32) * std::f32::consts::PI * 2.0 / n as f32;
            let radius = 0.3 + 0.2 * (i as f32 / n as f32);
            Vec2::new(
                0.5 + radius * angle.cos(),
                0.5 + radius * angle.sin(),
            )
        })
        .collect::<Vec<Vec2>>();

    // Velocities - make particles spiral outward
    let v = (0..n as u32)
        .map(|i| {
            let angle = (i as f32) * std::f32::consts::PI * 2.0 / n as f32;
            Vec2::new(
                0.001 * angle.cos(),
                0.001 * angle.sin(),
            )
        })
        .collect::<Vec<Vec2>>();

    println!("Created {} particles", n);

    // Create compute runner
    println!("Initializing Vulkan compute...");
    let runner = VulkanoRunner::new(shader_buffers)?;
    println!("Compute runner initialized!");

    // Create application state
    let mut app = App::new(runner, a, b, c, d, x, v);

    // Create event loop and run
    let event_loop = EventLoop::new()?;
    event_loop.run_app(&mut app)?;

    Ok(())
}
