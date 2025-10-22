//! Integration tests for VulkanoComputeChain
//!
//! These tests exercise the key workflows around VulkanoComputeChain:
//! - Setting up concrete data vecs that will provide the initial input to the compute chain
//! - Setting up buf_specs based on that data
//! - Setting up kernel specs
//! - Setting compute shader pipeline chains from vec of invoc_specs
//! - Setting up a VulkanoComputeChain with the needed buf_specs and invocation chain
//! - Executing the chain (1 or more times)
//! - Verifying that the correct outputs are stored in the buffers

use bytemuck::Zeroable;
use glam::Vec2;
use rust_gpu_chimera_demo::runners::{
    vulkano::{
        buffer_specs::buf_spec,
        shader_pipeline_builder::{invoc_spec, kernel},
    },
    vulkano_compute_chain::VulkanoComputeChain,
};
use shared::{grid::GridCell, num_workgroups_1d, num_workgroups_2d};

//
// BASIC SINGLE KERNEL TESTS
//

#[test]
fn test_single_adder_execution() {
    // Create data for the adder shader (a += b)
    let n = 64;
    let mut a = vec![1u32; n];
    let mut b = vec![10u32; n];

    // Create buf_specs
    let buf_specs = (buf_spec("a", 0, &mut a), buf_spec("b", 1, &mut b));

    // Create kernel configuration
    let wg_1d = num_workgroups_1d(n as u32);
    let adder_kernel = kernel("adder", vec![0, 1], wg_1d);

    // Create invocation chain with a single invocation
    let invocation_chain = vec![invoc_spec("adder_ab", vec!["a", "b"], adder_kernel)];

    // Create and execute the compute chain
    let compute_chain = VulkanoComputeChain::new(&buf_specs, invocation_chain)
        .expect("Failed to create VulkanoComputeChain");

    compute_chain.execute().expect("Failed to execute");

    // Verify results
    let buffer_a = compute_chain
        .typed_subbuffer_by_name::<u32>("a")
        .expect("Failed to get buffer a");
    let a_read = buffer_a.read().expect("Failed to read buffer a");

    // Expected: a[i] = 1 + 10 = 11
    let expected = vec![11u32; n];
    assert_eq!(
        &a_read[..],
        &expected[..],
        "Buffer a was not updated correctly"
    );
}

#[test]
fn test_single_step_particles_execution() {
    // Create data for the step_particles shader (x += v)
    let n = 64;
    let mut x = vec![Vec2::new(0.5, 0.5); n];
    let mut v = vec![Vec2::new(0.1, 0.2); n];

    // Create buf_specs
    let buf_specs = (buf_spec("x", 2, &mut x), buf_spec("v", 3, &mut v));

    // Create kernel configuration
    let wg_1d = num_workgroups_1d(n as u32);
    let step_particles_kernel = kernel("step_particles", vec![2, 3], wg_1d);

    // Create invocation chain
    let invocation_chain = vec![invoc_spec(
        "step_particles_0",
        vec!["x", "v"],
        step_particles_kernel,
    )];

    // Create and execute the compute chain
    let compute_chain = VulkanoComputeChain::new(&buf_specs, invocation_chain)
        .expect("Failed to create VulkanoComputeChain");

    compute_chain.execute().expect("Failed to execute");

    // Verify results
    let buffer_x = compute_chain
        .typed_subbuffer_by_name::<Vec2>("x")
        .expect("Failed to get buffer x");
    let x_read = buffer_x.read().expect("Failed to read buffer x");

    // Expected: x = 0.5 + 0.1 = 0.6, y = 0.5 + 0.2 = 0.7
    let expected = Vec2::new(0.6, 0.7);
    for (i, &actual) in x_read.iter().enumerate() {
        assert!(
            (actual.x - expected.x).abs() < 1e-5 && (actual.y - expected.y).abs() < 1e-5,
            "Position x[{}] incorrect. Got: {:?}, Expected: {:?}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_single_wrap_particles_execution() {
    // Create data for the wrap_particles shader (x = x % 1.0)
    let n = 64;
    // Start with some particles outside [0, 1] range
    let mut x = vec![Vec2::new(1.3, 2.7); n];

    // Create buf_specs
    let buf_specs = (buf_spec("x", 2, &mut x),);

    // Create kernel configuration
    let wg_1d = num_workgroups_1d(n as u32);
    let wrap_particles_kernel = kernel("wrap_particles", vec![2], wg_1d);

    // Create invocation chain
    let invocation_chain = vec![invoc_spec(
        "wrap_particles",
        vec!["x"],
        wrap_particles_kernel,
    )];

    // Create and execute the compute chain
    let compute_chain = VulkanoComputeChain::new(&buf_specs, invocation_chain)
        .expect("Failed to create VulkanoComputeChain");

    compute_chain.execute().expect("Failed to execute");

    // Verify results
    let buffer_x = compute_chain
        .typed_subbuffer_by_name::<Vec2>("x")
        .expect("Failed to get buffer x");
    let x_read = buffer_x.read().expect("Failed to read buffer x");

    // Expected: x = 1.3 % 1.0 = 0.3, y = 2.7 % 1.0 = 0.7
    let expected = Vec2::new(0.3, 0.7);
    for (i, &actual) in x_read.iter().enumerate() {
        assert!(
            (actual.x - expected.x).abs() < 1e-5 && (actual.y - expected.y).abs() < 1e-5,
            "Position x[{}] incorrect after wrapping. Got: {:?}, Expected: {:?}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_single_fill_grid_random_execution() {
    // Create data for the fill_grid_random shader
    let grid_size = 256;
    let mut grid = vec![GridCell::zeroed(); grid_size * grid_size];

    // Create buf_specs
    let buf_specs = (buf_spec("grid", 4, &mut grid),);

    // Create kernel configuration for 2D dispatch
    let wg_2d = num_workgroups_2d(grid_size as u32, grid_size as u32);
    let fill_grid_random_kernel = kernel("fill_grid_random", vec![4], wg_2d);

    // Create invocation chain
    let invocation_chain = vec![invoc_spec(
        "fill_grid_random",
        vec!["grid"],
        fill_grid_random_kernel,
    )];

    // Create and execute the compute chain
    let compute_chain = VulkanoComputeChain::new(&buf_specs, invocation_chain)
        .expect("Failed to create VulkanoComputeChain");

    compute_chain.execute().expect("Failed to execute");

    // Verify results
    let buffer_grid = compute_chain
        .typed_subbuffer_by_name::<GridCell>("grid")
        .expect("Failed to get buffer grid");
    let grid_read = buffer_grid.read().expect("Failed to read buffer grid");

    // Verify that the grid has been filled with non-zero values
    // (the shader fills with pseudo-random values)
    let mut non_zero_count = 0;
    for cell in grid_read.iter() {
        if cell.mass > 0.0 || cell.velocity.x != 0.0 || cell.velocity.y != 0.0 {
            non_zero_count += 1;
        }
    }

    // Expect most cells to have non-zero values (random, so not all will be zero)
    assert!(
        non_zero_count > (grid_size * grid_size) / 2,
        "Grid should be filled with mostly non-zero values. Found {} non-zero cells out of {}",
        non_zero_count,
        grid_size * grid_size
    );
}

//
// MULTIPLE EXECUTION TESTS
//

#[test]
fn test_multiple_adder_executions() {
    // Test executing the adder kernel multiple times with different buffers
    // NOTE: The adder kernel always uses bindings 0 and 1, but we can map different
    // logical buffer names to those bindings in different invocations
    let n = 64;
    let mut a = vec![0u32; n];
    let mut b = vec![1u32; n];
    let mut c = vec![10u32; n];
    let mut d = vec![100u32; n];

    // Create buf_specs - map buffers to bindings
    // The adder kernel expects binding 0 (first arg) and binding 1 (second arg)
    let buf_specs = (
        buf_spec("a", 0, &mut a),
        buf_spec("b", 1, &mut b),
        buf_spec("c", 1, &mut c), // c also maps to binding 1
        buf_spec("d", 1, &mut d), // d also maps to binding 1
    );

    // Create kernel configuration
    let wg_1d = num_workgroups_1d(n as u32);
    let adder_kernel = kernel("adder", vec![0, 1], wg_1d);

    // Create invocation chain: a += b, then a += c, then a += d
    // Each invocation maps buffer names to the shader's expected bindings
    let invocation_chain = vec![
        invoc_spec("adder_ab", vec!["a", "b"], adder_kernel.clone()),
        invoc_spec("adder_ac", vec!["a", "c"], adder_kernel.clone()),
        invoc_spec("adder_ad", vec!["a", "d"], adder_kernel.clone()),
    ];

    // Create and execute the compute chain
    let compute_chain = VulkanoComputeChain::new(&buf_specs, invocation_chain)
        .expect("Failed to create VulkanoComputeChain");

    compute_chain.execute().expect("Failed to execute");

    // Verify results
    let buffer_a = compute_chain
        .typed_subbuffer_by_name::<u32>("a")
        .expect("Failed to get buffer a");
    let a_read = buffer_a.read().expect("Failed to read buffer a");

    // Expected: a[i] = 0 + 1 + 10 + 100 = 111
    let expected = vec![111u32; n];
    assert_eq!(
        &a_read[..],
        &expected[..],
        "Buffer a was not updated correctly after multiple additions"
    );
}

#[test]
fn test_multiple_step_particles_executions() {
    // Test stepping particles multiple times
    let n = 64;
    let mut x = vec![Vec2::new(0.1, 0.2); n];
    let mut v = vec![Vec2::new(0.05, 0.1); n];

    // Create buf_specs
    let buf_specs = (buf_spec("x", 2, &mut x), buf_spec("v", 3, &mut v));

    // Create kernel configuration
    let wg_1d = num_workgroups_1d(n as u32);
    let step_particles_kernel = kernel("step_particles", vec![2, 3], wg_1d);

    // Create invocation chain: step 4 times
    let invocation_chain = vec![
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
    ];

    // Create and execute the compute chain
    let compute_chain = VulkanoComputeChain::new(&buf_specs, invocation_chain)
        .expect("Failed to create VulkanoComputeChain");

    compute_chain.execute().expect("Failed to execute");

    // Verify results
    let buffer_x = compute_chain
        .typed_subbuffer_by_name::<Vec2>("x")
        .expect("Failed to get buffer x");
    let x_read = buffer_x.read().expect("Failed to read buffer x");

    // Expected: x = 0.1 + 4*0.05 = 0.3, y = 0.2 + 4*0.1 = 0.6
    let expected = Vec2::new(0.3, 0.6);
    for (i, &actual) in x_read.iter().enumerate() {
        assert!(
            (actual.x - expected.x).abs() < 1e-5 && (actual.y - expected.y).abs() < 1e-5,
            "Position x[{}] incorrect after 4 steps. Got: {:?}, Expected: {:?}",
            i,
            actual,
            expected
        );
    }
}

//
// CHAINED KERNEL TESTS
//

#[test]
fn test_chained_step_and_wrap() {
    // Test chaining step_particles followed by wrap_particles
    let n = 64;
    let mut x = vec![Vec2::new(0.8, 0.9); n];
    let mut v = vec![Vec2::new(0.15, 0.15); n];

    // Create buf_specs
    let buf_specs = (buf_spec("x", 2, &mut x), buf_spec("v", 3, &mut v));

    // Create kernel configurations
    let wg_1d = num_workgroups_1d(n as u32);
    let step_particles_kernel = kernel("step_particles", vec![2, 3], wg_1d);
    let wrap_particles_kernel = kernel("wrap_particles", vec![2], wg_1d);

    // Create invocation chain: step twice, then wrap
    let invocation_chain = vec![
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
        invoc_spec("wrap_particles", vec!["x"], wrap_particles_kernel),
    ];

    // Create and execute the compute chain
    let compute_chain = VulkanoComputeChain::new(&buf_specs, invocation_chain)
        .expect("Failed to create VulkanoComputeChain");

    compute_chain.execute().expect("Failed to execute");

    // Verify results
    let buffer_x = compute_chain
        .typed_subbuffer_by_name::<Vec2>("x")
        .expect("Failed to get buffer x");
    let x_read = buffer_x.read().expect("Failed to read buffer x");

    // Expected: x = 0.8 + 2*0.15 = 1.1 -> wrap -> 0.1
    //           y = 0.9 + 2*0.15 = 1.2 -> wrap -> 0.2
    let expected = Vec2::new(0.1, 0.2);
    for (i, &actual) in x_read.iter().enumerate() {
        assert!(
            (actual.x - expected.x).abs() < 1e-5 && (actual.y - expected.y).abs() < 1e-5,
            "Position x[{}] incorrect after step+wrap. Got: {:?}, Expected: {:?}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_full_chain_adder_and_particles() {
    // Test a full chain combining adder and particle operations
    let n = 128;

    let mut a = vec![1u32; n];
    let mut b = vec![2u32; n];
    let mut x = vec![Vec2::new(0.5, 0.5); n];
    let mut v = vec![Vec2::new(0.1, 0.1); n];

    // Create buf_specs
    let buf_specs = (
        buf_spec("a", 0, &mut a),
        buf_spec("b", 1, &mut b),
        buf_spec("x", 2, &mut x),
        buf_spec("v", 3, &mut v),
    );

    // Create kernel configurations
    let wg_1d = num_workgroups_1d(n as u32);
    let adder_kernel = kernel("adder", vec![0, 1], wg_1d);
    let step_particles_kernel = kernel("step_particles", vec![2, 3], wg_1d);
    let wrap_particles_kernel = kernel("wrap_particles", vec![2], wg_1d);

    // Create invocation chain
    let invocation_chain = vec![
        invoc_spec("adder_ab", vec!["a", "b"], adder_kernel),
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
        invoc_spec("wrap_particles", vec!["x"], wrap_particles_kernel),
    ];

    // Create and execute the compute chain
    let compute_chain = VulkanoComputeChain::new(&buf_specs, invocation_chain)
        .expect("Failed to create VulkanoComputeChain");

    compute_chain.execute().expect("Failed to execute");

    // Verify adder results
    let buffer_a = compute_chain
        .typed_subbuffer_by_name::<u32>("a")
        .expect("Failed to get buffer a");
    let a_read = buffer_a.read().expect("Failed to read buffer a");
    assert_eq!(&a_read[..], &vec![3u32; n][..], "Adder result incorrect");

    // Verify particle results
    let buffer_x = compute_chain
        .typed_subbuffer_by_name::<Vec2>("x")
        .expect("Failed to get buffer x");
    let x_read = buffer_x.read().expect("Failed to read buffer x");

    // Expected: x = 0.5 + 2*0.1 = 0.7
    let expected = Vec2::new(0.7, 0.7);
    for (i, &actual) in x_read.iter().enumerate() {
        assert!(
            (actual.x - expected.x).abs() < 1e-5 && (actual.y - expected.y).abs() < 1e-5,
            "Position x[{}] incorrect. Got: {:?}, Expected: {:?}",
            i,
            actual,
            expected
        );
    }
}

//
// VARIED WORKGROUP SIZE TESTS
//

#[test]
fn test_large_buffer_multiple_workgroups() {
    // Test with a buffer size that requires multiple workgroups
    let n = 512; // Requires 8 workgroups with WORKGROUP_SIZE=64

    let mut a = vec![5u32; n];
    let mut b = vec![3u32; n];

    // Create buf_specs
    let buf_specs = (buf_spec("a", 0, &mut a), buf_spec("b", 1, &mut b));

    // Create kernel configuration
    let wg_1d = num_workgroups_1d(n as u32);
    let adder_kernel = kernel("adder", vec![0, 1], wg_1d);

    // Create invocation chain
    let invocation_chain = vec![invoc_spec("adder_ab", vec!["a", "b"], adder_kernel)];

    // Create and execute the compute chain
    let compute_chain = VulkanoComputeChain::new(&buf_specs, invocation_chain)
        .expect("Failed to create VulkanoComputeChain");

    compute_chain.execute().expect("Failed to execute");

    // Verify results
    let buffer_a = compute_chain
        .typed_subbuffer_by_name::<u32>("a")
        .expect("Failed to get buffer a");
    let a_read = buffer_a.read().expect("Failed to read buffer a");

    // Expected: a[i] = 5 + 3 = 8
    let expected = vec![8u32; n];
    assert_eq!(
        &a_read[..],
        &expected[..],
        "Large buffer not computed correctly"
    );
}

#[test]
fn test_large_2d_grid() {
    // Test with a large 2D grid
    let grid_size = 128; // 128x128 grid
    let mut grid = vec![GridCell::zeroed(); grid_size * grid_size];

    // Create buf_specs
    let buf_specs = (buf_spec("grid", 4, &mut grid),);

    // Create kernel configuration for 2D dispatch
    let wg_2d = num_workgroups_2d(grid_size as u32, grid_size as u32);
    let fill_grid_random_kernel = kernel("fill_grid_random", vec![4], wg_2d);

    // Create invocation chain
    let invocation_chain = vec![invoc_spec(
        "fill_grid_random",
        vec!["grid"],
        fill_grid_random_kernel,
    )];

    // Create and execute the compute chain
    let compute_chain = VulkanoComputeChain::new(&buf_specs, invocation_chain)
        .expect("Failed to create VulkanoComputeChain");

    compute_chain.execute().expect("Failed to execute");

    // Verify results
    let buffer_grid = compute_chain
        .typed_subbuffer_by_name::<GridCell>("grid")
        .expect("Failed to get buffer grid");
    let grid_read = buffer_grid.read().expect("Failed to read buffer grid");

    // Verify grid size is correct
    assert_eq!(
        grid_read.len(),
        grid_size * grid_size,
        "Grid size incorrect"
    );

    // Sample a few cells to verify they're filled
    assert!(
        grid_read[0].mass >= 0.0 && grid_read[0].mass <= 1.0,
        "Grid cell mass out of expected range"
    );
}

//
// EDGE CASE TESTS
//

#[test]
fn test_zero_inputs() {
    // Test with all zero inputs
    let n = 64;
    let mut a = vec![0u32; n];
    let mut b = vec![0u32; n];

    // Create buf_specs
    let buf_specs = (buf_spec("a", 0, &mut a), buf_spec("b", 1, &mut b));

    // Create kernel configuration
    let wg_1d = num_workgroups_1d(n as u32);
    let adder_kernel = kernel("adder", vec![0, 1], wg_1d);

    // Create invocation chain
    let invocation_chain = vec![invoc_spec("adder_ab", vec!["a", "b"], adder_kernel)];

    // Create and execute the compute chain
    let compute_chain = VulkanoComputeChain::new(&buf_specs, invocation_chain)
        .expect("Failed to create VulkanoComputeChain");

    compute_chain.execute().expect("Failed to execute");

    // Verify results
    let buffer_a = compute_chain
        .typed_subbuffer_by_name::<u32>("a")
        .expect("Failed to get buffer a");
    let a_read = buffer_a.read().expect("Failed to read buffer a");

    // Expected: all zeros
    let expected = vec![0u32; n];
    assert_eq!(&a_read[..], &expected[..], "Zero input test failed");
}

#[test]
fn test_overflow_behavior() {
    // Test u32 overflow behavior
    let n = 64;
    let mut a = vec![u32::MAX; n];
    let mut b = vec![1u32; n];

    // Create buf_specs
    let buf_specs = (buf_spec("a", 0, &mut a), buf_spec("b", 1, &mut b));

    // Create kernel configuration
    let wg_1d = num_workgroups_1d(n as u32);
    let adder_kernel = kernel("adder", vec![0, 1], wg_1d);

    // Create invocation chain
    let invocation_chain = vec![invoc_spec("adder_ab", vec!["a", "b"], adder_kernel)];

    // Create and execute the compute chain
    let compute_chain = VulkanoComputeChain::new(&buf_specs, invocation_chain)
        .expect("Failed to create VulkanoComputeChain");

    compute_chain.execute().expect("Failed to execute");

    // Verify results
    let buffer_a = compute_chain
        .typed_subbuffer_by_name::<u32>("a")
        .expect("Failed to get buffer a");
    let a_read = buffer_a.read().expect("Failed to read buffer a");

    // Expected: u32::MAX + 1 wraps to 0
    let expected = vec![0u32; n];
    assert_eq!(&a_read[..], &expected[..], "Overflow test failed");
}

#[test]
fn test_negative_velocity_wrapping() {
    // Test particles with negative velocities that go below 0
    let n = 64;
    let mut x = vec![Vec2::new(0.1, 0.2); n];
    let mut v = vec![Vec2::new(-0.05, -0.1); n];

    // Create buf_specs
    let buf_specs = (buf_spec("x", 2, &mut x), buf_spec("v", 3, &mut v));

    // Create kernel configurations
    let wg_1d = num_workgroups_1d(n as u32);
    let step_particles_kernel = kernel("step_particles", vec![2, 3], wg_1d);
    let wrap_particles_kernel = kernel("wrap_particles", vec![2], wg_1d);

    // Create invocation chain
    let invocation_chain = vec![
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
        invoc_spec("wrap_particles", vec!["x"], wrap_particles_kernel),
    ];

    // Create and execute the compute chain
    let compute_chain = VulkanoComputeChain::new(&buf_specs, invocation_chain)
        .expect("Failed to create VulkanoComputeChain");

    compute_chain.execute().expect("Failed to execute");

    // Verify results
    let buffer_x = compute_chain
        .typed_subbuffer_by_name::<Vec2>("x")
        .expect("Failed to get buffer x");
    let x_read = buffer_x.read().expect("Failed to read buffer x");

    // Expected: x = 0.1 + 2*(-0.05) = 0.0
    //           y = 0.2 + 2*(-0.1) = 0.0
    let expected = Vec2::new(0.0, 0.0);
    for (i, &actual) in x_read.iter().enumerate() {
        assert!(
            (actual.x - expected.x).abs() < 1e-5 && (actual.y - expected.y).abs() < 1e-5,
            "Position x[{}] incorrect with negative velocity. Got: {:?}, Expected: {:?}",
            i,
            actual,
            expected
        );
    }
}

//
// EXECUTION MULTIPLE TIMES TESTS
//

#[test]
fn test_execute_chain_multiple_times() {
    // Test executing the same chain multiple times
    let n = 64;
    let mut a = vec![0u32; n];
    let mut b = vec![1u32; n];

    // Create buf_specs
    let buf_specs = (buf_spec("a", 0, &mut a), buf_spec("b", 1, &mut b));

    // Create kernel configuration
    let wg_1d = num_workgroups_1d(n as u32);
    let adder_kernel = kernel("adder", vec![0, 1], wg_1d);

    // Create invocation chain
    let invocation_chain = vec![invoc_spec("adder_ab", vec!["a", "b"], adder_kernel)];

    // Create the compute chain
    let compute_chain = VulkanoComputeChain::new(&buf_specs, invocation_chain)
        .expect("Failed to create VulkanoComputeChain");

    // Execute 3 times
    compute_chain.execute().expect("Failed to execute (1)");
    compute_chain.execute().expect("Failed to execute (2)");
    compute_chain.execute().expect("Failed to execute (3)");

    // Verify results
    let buffer_a = compute_chain
        .typed_subbuffer_by_name::<u32>("a")
        .expect("Failed to get buffer a");
    let a_read = buffer_a.read().expect("Failed to read buffer a");

    // Expected: a = 0 + 1 + 1 + 1 = 3 (executed 3 times)
    let expected = vec![3u32; n];
    assert_eq!(&a_read[..], &expected[..], "Multiple executions failed");
}

#[test]
fn test_varied_data_patterns() {
    // Test with varied input data (not all uniform)
    let n = 64;
    let mut a = (0..n as u32).collect::<Vec<u32>>();
    let mut b = (0..n as u32).map(|i| i * 2).collect::<Vec<u32>>();

    // Create buf_specs
    let buf_specs = (buf_spec("a", 0, &mut a), buf_spec("b", 1, &mut b));

    // Create kernel configuration
    let wg_1d = num_workgroups_1d(n as u32);
    let adder_kernel = kernel("adder", vec![0, 1], wg_1d);

    // Create invocation chain
    let invocation_chain = vec![invoc_spec("adder_ab", vec!["a", "b"], adder_kernel)];

    // Create and execute the compute chain
    let compute_chain = VulkanoComputeChain::new(&buf_specs, invocation_chain)
        .expect("Failed to create VulkanoComputeChain");

    compute_chain.execute().expect("Failed to execute");

    // Verify results
    let buffer_a = compute_chain
        .typed_subbuffer_by_name::<u32>("a")
        .expect("Failed to get buffer a");
    let a_read = buffer_a.read().expect("Failed to read buffer a");

    // Expected: a[i] = i + 2*i = 3*i
    let expected: Vec<u32> = (0..n as u32).map(|i| 3 * i).collect();
    assert_eq!(
        &a_read[..],
        &expected[..],
        "Varied data pattern test failed"
    );
}
