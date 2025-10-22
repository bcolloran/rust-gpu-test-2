// Integration tests for the complete compute pass
//
// These tests verify the `run_compute_and_get_buffer` function which:
// 1. Runs the adder shader 3 times (a += b, a += c, a += d)
// 2. Runs step_particles 4 times (x += v, repeated 4 times)
// 3. Runs wrap_particles once (x = x % 1.0)
//
// The tests verify that all buffers are correctly modified after the pass.
