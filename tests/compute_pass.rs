// Integration tests for the complete compute pass
//
// These tests verify the `run_compute_and_get_buffer` function which:
// 1. Runs the adder shader 3 times (a += b, a += c, a += d)
// 2. Runs step_particles 4 times (x += v, repeated 4 times)
// 3. Runs wrap_particles once (x = x % 1.0)
//
// The tests verify that all buffers are correctly modified after the pass.

use bytemuck::Zeroable;
use glam::Vec2;
use rust_gpu_chimera_demo::runners::vulkano::shader_buffer_mapping::ComputePassInvocationInfo;
use rust_gpu_chimera_demo::runners::vulkano::VulkanoRunner;
use shared::grid::GridCell;

/// Create a VulkanoRunner with the full compute pass configuration
/// This matches the configuration used in main.rs
fn create_full_compute_pass_runner() -> VulkanoRunner {
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

    VulkanoRunner::new(shader_buffers).expect("Failed to create VulkanoRunner")
}

#[test]
fn test_compute_pass_basic() {
    let runner = create_full_compute_pass_runner();

    // Simple input data
    let mut a = vec![1u32; 8];
    let b = vec![10u32; 8];
    let c = vec![100u32; 8];
    let d = vec![1000u32; 8];

    let mut x = vec![Vec2::new(0.5, 0.5); 8];
    let v = vec![Vec2::new(0.1, 0.1); 8];

    let n = 256;
    let mut g = (0..(n * n)).map(|_| GridCell::zeroed()).collect::<Vec<_>>();

    // Expected results:
    // a = 1 + 10 + 100 + 1000 = 1111
    let expected_a = vec![1111u32; 8];

    // x = 0.5 + 4 * 0.1 = 0.9 (step_particles runs 4 times)
    // Then wrap (x % 1.0) = 0.9
    let expected_x = vec![Vec2::new(0.9, 0.9); 8];

    // Run the compute pass
    let result = runner.run_compute_and_get_buffer(&mut a, &b, &c, &d, &mut x, &v, &mut g);
    assert!(result.is_ok(), "Compute pass failed: {:?}", result.err());

    // Verify buffer a
    assert_eq!(
        a, expected_a,
        "Buffer a not computed correctly. Got: {:?}, Expected: {:?}",
        a, expected_a
    );

    // Verify buffer x (with some tolerance for floating point)
    for (i, (actual, expected)) in x.iter().zip(expected_x.iter()).enumerate() {
        assert!(
            (actual.x - expected.x).abs() < 1e-5 && (actual.y - expected.y).abs() < 1e-5,
            "Buffer x[{}] not computed correctly. Got: {:?}, Expected: {:?}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_compute_pass_wrapping() {
    let runner = create_full_compute_pass_runner();

    let mut a = vec![0u32; 8];
    let b = vec![0u32; 8];
    let c = vec![0u32; 8];
    let d = vec![0u32; 8];

    // Start particles near the edge so they wrap
    let mut x = vec![Vec2::new(0.95, 0.95); 8];
    let v = vec![Vec2::new(0.05, 0.05); 8];

    let n = 256;
    let mut g = (0..(n * n)).map(|_| GridCell::zeroed()).collect::<Vec<_>>();

    // x = 0.95 + 4 * 0.05 = 1.15
    // After wrap: x % 1.0 = 0.15
    let expected_x = vec![Vec2::new(0.15, 0.15); 8];

    let result = runner.run_compute_and_get_buffer(&mut a, &b, &c, &d, &mut x, &v, &mut g);
    assert!(result.is_ok(), "Compute pass failed: {:?}", result.err());

    // Verify wrapping occurred correctly
    for (i, (actual, expected)) in x.iter().zip(expected_x.iter()).enumerate() {
        assert!(
            (actual.x - expected.x).abs() < 1e-5 && (actual.y - expected.y).abs() < 1e-5,
            "Buffer x[{}] wrapping incorrect. Got: {:?}, Expected: {:?}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_compute_pass_multiple_wraps() {
    let runner = create_full_compute_pass_runner();

    let mut a = vec![0u32; 8];
    let b = vec![0u32; 8];
    let c = vec![0u32; 8];
    let d = vec![0u32; 8];

    // Large velocities that wrap multiple times
    let mut x = vec![Vec2::new(0.5, 0.5); 8];
    let v = vec![Vec2::new(0.3, 0.4); 8];

    let n = 256;
    let mut g = (0..(n * n)).map(|_| GridCell::zeroed()).collect::<Vec<_>>();

    // x.x = 0.5 + 4 * 0.3 = 1.7, wrapped = 0.7
    // x.y = 0.5 + 4 * 0.4 = 2.1, wrapped = 0.1 (wraps twice)
    let expected_x = vec![Vec2::new(0.7, 0.1); 8];

    let result = runner.run_compute_and_get_buffer(&mut a, &b, &c, &d, &mut x, &v, &mut g);
    assert!(result.is_ok(), "Compute pass failed: {:?}", result.err());

    for (i, (actual, expected)) in x.iter().zip(expected_x.iter()).enumerate() {
        assert!(
            (actual.x - expected.x).abs() < 1e-5 && (actual.y - expected.y).abs() < 1e-5,
            "Buffer x[{}] multiple wrapping incorrect. Got: {:?}, Expected: {:?}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_compute_pass_varied_data() {
    let runner = create_full_compute_pass_runner();

    // Use varied input data
    let mut a = (0..8u32).collect::<Vec<u32>>();
    let b = (0..8u32).map(|i| i * 2).collect::<Vec<u32>>();
    let c = (0..8u32).map(|i| i * 3).collect::<Vec<u32>>();
    let d = (0..8u32).map(|i| i * 4).collect::<Vec<u32>>();

    // a[i] = i + 2*i + 3*i + 4*i = 10*i
    let expected_a: Vec<u32> = (0..8u32).map(|i| i * 10).collect();

    let mut x = vec![
        Vec2::new(0.1, 0.2),
        Vec2::new(0.2, 0.3),
        Vec2::new(0.3, 0.4),
        Vec2::new(0.4, 0.5),
        Vec2::new(0.5, 0.6),
        Vec2::new(0.6, 0.7),
        Vec2::new(0.7, 0.8),
        Vec2::new(0.8, 0.9),
    ];
    let v = vec![
        Vec2::new(0.01, 0.02),
        Vec2::new(0.02, 0.03),
        Vec2::new(0.03, 0.04),
        Vec2::new(0.04, 0.05),
        Vec2::new(0.05, 0.06),
        Vec2::new(0.06, 0.07),
        Vec2::new(0.07, 0.08),
        Vec2::new(0.08, 0.09),
    ];
    let n = 256;
    let mut g = (0..(n * n)).map(|_| GridCell::zeroed()).collect::<Vec<_>>();
    // Calculate expected x values
    let expected_x: Vec<Vec2> = x
        .iter()
        .zip(v.iter())
        .map(|(xi, vi)| {
            let new_x = *xi + *vi * 4.0; // 4 step_particles calls
            Vec2::new(new_x.x % 1.0, new_x.y % 1.0) // wrap_particles
        })
        .collect();

    let result = runner.run_compute_and_get_buffer(&mut a, &b, &c, &d, &mut x, &v, &mut g);
    assert!(result.is_ok(), "Compute pass failed: {:?}", result.err());

    // Verify buffer a
    assert_eq!(
        a, expected_a,
        "Buffer a not computed correctly with varied data"
    );

    // Verify buffer x
    for (i, (actual, expected)) in x.iter().zip(expected_x.iter()).enumerate() {
        assert!(
            (actual.x - expected.x).abs() < 1e-5 && (actual.y - expected.y).abs() < 1e-5,
            "Buffer x[{}] not computed correctly. Got: {:?}, Expected: {:?}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_compute_pass_negative_velocities() {
    let runner = create_full_compute_pass_runner();

    let mut a = vec![0u32; 8];
    let b = vec![0u32; 8];
    let c = vec![0u32; 8];
    let d = vec![0u32; 8];

    // Negative velocities (particles moving backwards)
    let mut x = vec![Vec2::new(0.5, 0.5); 8];
    let v = vec![Vec2::new(-0.05, -0.1); 8];

    let n = 256;
    let mut g = (0..(n * n)).map(|_| GridCell::zeroed()).collect::<Vec<_>>();

    // x = 0.5 + 4 * (-0.05, -0.1) = (0.3, 0.1)
    // After wrap: (0.3, 0.1) - both positive, no wrapping needed
    let expected_x = vec![Vec2::new(0.3, 0.1); 8];

    let result = runner.run_compute_and_get_buffer(&mut a, &b, &c, &d, &mut x, &v, &mut g);
    assert!(result.is_ok(), "Compute pass failed: {:?}", result.err());

    for (i, (actual, expected)) in x.iter().zip(expected_x.iter()).enumerate() {
        assert!(
            (actual.x - expected.x).abs() < 1e-5 && (actual.y - expected.y).abs() < 1e-5,
            "Buffer x[{}] negative velocity incorrect. Got: {:?}, Expected: {:?}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_compute_pass_large_buffer() {
    let runner = create_full_compute_pass_runner();

    // Use a larger buffer (multiple workgroups)
    let n = 512;

    let mut a = vec![1u32; n];
    let b = vec![2u32; n];
    let c = vec![4u32; n];
    let d = vec![8u32; n];

    // a = 1 + 2 + 4 + 8 = 15
    let expected_a = vec![15u32; n];

    let mut x = vec![Vec2::new(0.0, 0.0); n];
    let v = vec![Vec2::new(0.25, 0.25); n];

    let n = 256;
    let mut g = (0..(n * n)).map(|_| GridCell::zeroed()).collect::<Vec<_>>();

    // x = 0 + 4 * 0.25 = 1.0
    // After wrap: 1.0 % 1.0 = 0.0
    let expected_x = vec![Vec2::new(0.0, 0.0); n];

    let result = runner.run_compute_and_get_buffer(&mut a, &b, &c, &d, &mut x, &v, &mut g);
    assert!(result.is_ok(), "Compute pass failed: {:?}", result.err());

    assert_eq!(&a, &expected_a, "Buffer a incorrect for large buffer");

    for (i, (actual, expected)) in x.iter().zip(expected_x.iter()).enumerate() {
        assert!(
            (actual.x - expected.x).abs() < 1e-5 && (actual.y - expected.y).abs() < 1e-5,
            "Buffer x[{}] incorrect for large buffer. Got: {:?}, Expected: {:?}",
            i,
            actual,
            expected
        );
    }
}

#[test]
fn test_compute_pass_returns_buffer() {
    let runner = create_full_compute_pass_runner();

    let mut a = vec![0u32; 8];
    let b = vec![0u32; 8];
    let c = vec![0u32; 8];
    let d = vec![0u32; 8];

    let mut x = vec![Vec2::new(0.5, 0.5); 8];
    let v = vec![Vec2::new(0.0, 0.0); 8];

    let n = 256;
    let mut g = (0..(n * n)).map(|_| GridCell::zeroed()).collect::<Vec<_>>();

    let result = runner.run_compute_and_get_buffer(&mut a, &b, &c, &d, &mut x, &v, &mut g);
    assert!(result.is_ok(), "Compute pass failed");

    // Verify that the function returns the buffer and correct length
    let (buffer, len) = result.unwrap();
    assert_eq!(len, 8, "Returned length incorrect");

    // The buffer should be a valid Vulkan buffer
    // We can't easily verify the contents directly, but we can verify
    // that it's not null and has the right type signature
    assert!(buffer.size() > 0, "Returned buffer has zero size");
}

#[test]
fn test_compute_pass_zero_input() {
    let runner = create_full_compute_pass_runner();

    // All zeros
    let mut a = vec![0u32; 8];
    let b = vec![0u32; 8];
    let c = vec![0u32; 8];
    let d = vec![0u32; 8];

    let mut x = vec![Vec2::ZERO; 8];
    let v = vec![Vec2::ZERO; 8];

    let n = 256;
    let mut g = (0..(n * n)).map(|_| GridCell::zeroed()).collect::<Vec<_>>();

    let expected_a = vec![0u32; 8];
    let expected_x = vec![Vec2::ZERO; 8];

    let result = runner.run_compute_and_get_buffer(&mut a, &b, &c, &d, &mut x, &v, &mut g);
    assert!(result.is_ok(), "Compute pass failed with zero input");

    assert_eq!(a, expected_a, "Buffer a incorrect for zero input");
    assert_eq!(x, expected_x, "Buffer x incorrect for zero input");
}

#[test]
fn test_compute_pass_max_values() {
    let runner = create_full_compute_pass_runner();

    // Test with maximum u32 values (should wrap)
    let mut a = vec![u32::MAX; 8];
    let b = vec![1u32; 8];
    let c = vec![0u32; 8];
    let d = vec![0u32; 8];

    // a = u32::MAX + 1 + 0 + 0 = 0 (wraps around)
    let expected_a = vec![0u32; 8];

    let mut x = vec![Vec2::new(0.5, 0.5); 8];
    let v = vec![Vec2::new(0.0, 0.0); 8];

    let n = 256;
    let mut g = (0..(n * n)).map(|_| GridCell::zeroed()).collect::<Vec<_>>();

    let result = runner.run_compute_and_get_buffer(&mut a, &b, &c, &d, &mut x, &v, &mut g);
    assert!(result.is_ok(), "Compute pass failed with max values");

    assert_eq!(a, expected_a, "Buffer a incorrect with overflow");
}

#[test]
fn test_compute_pass_spiral_pattern() {
    let runner = create_full_compute_pass_runner();

    let n = 256;

    let mut a = vec![0u32; n];
    let b = vec![0u32; n];
    let c = vec![0u32; n];
    let d = vec![0u32; n];

    // Create a spiral pattern similar to the main program
    let mut x: Vec<Vec2> = (0..n)
        .map(|i| {
            let angle = (i as f32) * std::f32::consts::PI * 2.0 / n as f32;
            let radius = 0.3;
            Vec2::new(0.5 + radius * angle.cos(), 0.5 + radius * angle.sin())
        })
        .collect();

    let v: Vec<Vec2> = (0..n)
        .map(|i| {
            let angle = (i as f32) * std::f32::consts::PI * 2.0 / n as f32;
            Vec2::new(0.01 * angle.cos(), 0.01 * angle.sin())
        })
        .collect();
    let n = 256;
    let mut g = (0..(n * n)).map(|_| GridCell::zeroed()).collect::<Vec<_>>();
    // Calculate expected values
    let expected_x: Vec<Vec2> = x
        .iter()
        .zip(v.iter())
        .map(|(xi, vi)| {
            let new_x = *xi + *vi * 4.0;
            Vec2::new(new_x.x % 1.0, new_x.y % 1.0)
        })
        .collect();

    let result = runner.run_compute_and_get_buffer(&mut a, &b, &c, &d, &mut x, &v, &mut g);
    assert!(result.is_ok(), "Compute pass failed with spiral pattern");

    // Verify the spiral pattern was computed correctly
    for (i, (actual, expected)) in x.iter().zip(expected_x.iter()).enumerate() {
        assert!(
            (actual.x - expected.x).abs() < 1e-4 && (actual.y - expected.y).abs() < 1e-4,
            "Buffer x[{}] spiral pattern incorrect. Got: {:?}, Expected: {:?}, Diff: ({}, {})",
            i,
            actual,
            expected,
            (actual.x - expected.x).abs(),
            (actual.y - expected.y).abs()
        );
    }
}
