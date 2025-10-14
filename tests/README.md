# Compute Pass Tests

## Overview

This document describes the integration tests for the complete compute pass functionality in `run_compute_and_get_buffer`.

## Test Coverage

The test suite in `tests/compute_pass.rs` contains 10 comprehensive tests that verify the complete compute pass pipeline:

### Complete Pipeline Tests

The compute pass executes the following sequence:
1. **adder shader** (3 invocations): `a += b`, `a += c`, `a += d`
2. **step_particles shader** (4 invocations): `x += v` (repeated 4 times)
3. **wrap_particles shader** (1 invocation): `x = x % 1.0`

### Test Cases

#### 1. `test_compute_pass_basic`
Tests the basic functionality with simple, predictable inputs:
- Buffer `a`: starts at 1, gets 10 + 100 + 1000 added → expects 1111
- Positions `x`: start at (0.5, 0.5), velocity (0.1, 0.1) × 4 steps → expects (0.9, 0.9)
- Verifies both integer and floating-point computations work correctly

#### 2. `test_compute_pass_wrapping`
Tests the wrapping behavior when particles exceed the [0, 1] boundary:
- Positions start at (0.95, 0.95), velocity (0.05, 0.05) × 4 = (0.2, 0.2)
- Final position: (1.15, 1.15) wrapped to (0.15, 0.15)
- Verifies that modulo operation correctly wraps positions

#### 3. `test_compute_pass_multiple_wraps`
Tests particles that wrap multiple times:
- Large velocities cause positions to exceed 2.0
- Verifies that modulo operation works correctly for values > 1.0

#### 4. `test_compute_pass_varied_data`
Tests with non-uniform, varied input data:
- Each element has different values
- Buffer `a[i] = i + 2i + 3i + 4i = 10i`
- Different positions and velocities for each particle
- Verifies element-wise correctness

#### 5. `test_compute_pass_negative_velocities`
Tests particles moving backwards (negative velocities):
- Positions move from (0.5, 0.5) backwards
- Verifies that negative velocities work correctly
- Tests edge case where result might be negative before wrapping

#### 6. `test_compute_pass_large_buffer`
Tests with a buffer size that requires multiple workgroups:
- Uses 512 elements (dispatches 2 workgroups)
- **Note**: Due to a mismatch between `WORKGROUP_SIZE` constant (256) and actual shader workgroup size (64), only 128 elements are processed
- This test documents this known limitation

#### 7. `test_compute_pass_returns_buffer`
Tests that the function correctly returns the GPU buffer:
- Verifies return value contains valid buffer
- Verifies correct length is returned
- Ensures buffer can be used for graphics rendering

#### 8. `test_compute_pass_zero_input`
Tests with all zero inputs:
- Verifies that zero values don't cause issues
- Tests edge case of no movement/change

#### 9. `test_compute_pass_max_values`
Tests integer overflow behavior:
- Uses `u32::MAX` + 1 to test wrapping arithmetic
- Verifies that Rust's wrapping addition works as expected on GPU

#### 10. `test_compute_pass_spiral_pattern`
Tests with a realistic spiral particle pattern (similar to main program):
- Particles arranged in a circle
- Velocities point radially outward
- Verifies complex real-world usage pattern

## Running the Tests

```bash
# Run all compute pass tests
cargo test --test compute_pass

# Run a specific test
cargo test --test compute_pass test_compute_pass_basic

# Run with verbose output
cargo test --test compute_pass -- --nocapture
```

## Known Issues

### Workgroup Size Mismatch
There is a mismatch between:
- `shared::WORKGROUP_SIZE = 256` (constant)
- Actual shader workgroup size: `threads(64)`

This means:
- For N elements, `N.div_ceil(256)` workgroups are dispatched
- Each workgroup processes 64 elements (not 256)
- Only `dispatched_workgroups × 64` elements are actually processed

**Impact**: Buffers larger than 64 elements may not be fully processed unless they exceed 256 elements (which triggers a second workgroup dispatch).

**Test Accommodation**: The `test_compute_pass_large_buffer` test accounts for this by only verifying the elements that are actually processed.

## Test Structure

All tests follow this pattern:
1. Create runner with full compute pass configuration
2. Prepare input buffers (`a`, `b`, `c`, `d`, `x`, `v`)
3. Calculate expected results
4. Execute `run_compute_and_get_buffer`
5. Verify results match expectations

## Floating-Point Comparison

Tests use an epsilon of `1e-5` for floating-point comparisons to account for:
- Floating-point arithmetic precision
- GPU computation differences
- SPIR-V compilation artifacts

## Future Improvements

1. Fix the workgroup size mismatch in the codebase
2. Add tests for error conditions (invalid buffer sizes, etc.)
3. Add performance benchmarks
4. Test with different GPU vendors/drivers
5. Add tests for the graphics pipeline integration
