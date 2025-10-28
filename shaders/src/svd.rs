use spirv_std::glam::{Mat2, Vec2};

#[cfg(not(test))]
use spirv_std::num_traits::float::Float;

#[derive(Clone, Copy, Debug)]
pub struct Svd2 {
    pub u: Mat2,
    pub s: Vec2, // (s1, s2) with s1 >= s2 >= 0
    pub v: Mat2,
}

#[inline(always)]
fn hypot2(x: f32, y: f32) -> f32 {
    // scale-safe sqrt(x^2 + y^2) in f32
    let ax = x.abs();
    let ay = y.abs();
    let (m, n) = if ax >= ay { (ax, ay) } else { (ay, ax) };
    if m == 0.0 {
        0.0
    } else {
        m * (1.0 + (n / m) * (n / m)).sqrt()
    }
}

/// Exact, stable 2x2 SVD for glam::Mat2 (f32).
/// Returns A = U * diag(s) * V^T with s.x >= s.y >= 0.
#[inline(always)]
pub fn svd2x2_exact(a: Mat2) -> Svd2 {
    // Unpack (glam is column-major)
    let a11 = a.x_axis.x;
    let a12 = a.y_axis.x;
    let a21 = a.x_axis.y;
    let a22 = a.y_axis.y;

    // Compute det(A) early - we'll need it for stable lambda2 computation
    let det_a = a11 * a22 - a12 * a21;

    // S = A^T A = [[α, β], [β, γ]]
    let alpha = a11 * a11 + a21 * a21;
    let beta = a11 * a12 + a21 * a22;
    let gamma = a12 * a12 + a22 * a22;

    // Eigenvalues of S (sorted): λ1 >= λ2 >= 0
    let g = alpha - gamma;
    let h = 2.0 * beta;
    let r = hypot2(g, h);
    let lambda1 = 0.5 * ((alpha + gamma) + r).max(0.0);

    // Compute lambda2 stably: use det(S) = det(A)^2 = lambda1 * lambda2 to avoid cancellation
    // This is much more stable than 0.5 * ((alpha + gamma) - r) when r ≈ (alpha + gamma)
    let det_a_sq = det_a * det_a;
    // Use a small epsilon to avoid division by zero; if lambda1 is tiny, lambda2 should be 0
    let lambda1_safe = lambda1.max(f32::MIN_POSITIVE);
    let lambda2 = (det_a_sq / lambda1_safe).max(0.0); // Stable eigenvector for λ1
                                                      // Use the more stable pair among [β, λ1-α] and [λ1-γ, β]
    let x1 = beta;
    let y1 = lambda1 - alpha;
    let u1 = lambda1 - gamma;
    let v1 = beta;

    let mut v0 = if x1.abs() > y1.abs() {
        Vec2::new(x1, y1)
    } else {
        Vec2::new(u1, v1)
    };
    // Handle pathological zero vector (perfectly diagonal S with α >= γ)
    if v0.length_squared() == 0.0 {
        v0 = if alpha >= gamma {
            Vec2::new(1.0, 0.0)
        } else {
            Vec2::new(0.0, 1.0)
        };
    } else {
        v0 = v0.normalize();
    }
    // Second right singular vector: orthogonal complement
    let v1 = Vec2::new(-v0.y, v0.x);
    let mut v_mat = Mat2::from_cols(v0, v1);

    // Ensure first column of V corresponds to the larger eigenvalue (λ1).
    // We can check by evaluating diag entries of V^T S V cheaply.
    let c0 = v_mat.x_axis;
    let c1 = v_mat.y_axis;
    let d11 = alpha * (c0.x * c0.x) + 2.0 * beta * (c0.x * c0.y) + gamma * (c0.y * c0.y);
    let d22 = alpha * (c1.x * c1.x) + 2.0 * beta * (c1.x * c1.y) + gamma * (c1.y * c1.y);
    if d22 > d11 {
        v_mat = Mat2::from_cols(c1, c0); // swap columns so column 0 ↔ λ1
    }

    // Singular values
    let s1 = lambda1.sqrt();
    let mut s2 = lambda2.sqrt();

    // B = A * V
    let b_mat = a * v_mat;

    // Left singular vectors: U = B * Σ^{-1}, with rank-aware handling
    let mut u0 = if s1 > 0.0 {
        b_mat.x_axis / s1
    } else {
        Vec2::new(1.0, 0.0)
    };

    // Robust rank-1 guard:
    // If det(A) == 0 exactly or s2 is denormally small,
    // pick a perpendicular for u1 instead of dividing noisy b_mat.y by ~0.
    let s2_tiny = s2 <= f32::MIN_POSITIVE * 8.0; // extremely tiny (denormal-ish)
    let mut u1 = if s2 == 0.0 || det_a == 0.0 || s2_tiny {
        s2 = 0.0; // lock it to zero in the rank-1 path
        Vec2::new(-u0.y, u0.x)
    } else {
        b_mat.y_axis / s2
    };

    // Orthonormalize defensively (harmless when already orthonormal).
    // This helps when s2 << s1 and tiny residuals in V^T S V are magnified.
    // One symmetric step is enough in 2D.
    let dot = u0.dot(u1);
    u1 = (u1 - u0 * dot).normalize();
    u0 = u0.normalize();

    let mut u_mat = Mat2::from_cols(u0, u1);

    // Optional: make det(U) >= 0 by flipping the second columns of U and V together.
    // This keeps A = U Σ V^T unchanged and preserves det(U)*det(V) = sign(det(A)).
    if u_mat.determinant() < 0.0 {
        u_mat = Mat2::from_cols(u_mat.x_axis, -u_mat.y_axis);
        v_mat = Mat2::from_cols(v_mat.x_axis, -v_mat.y_axis);
    }

    Svd2 {
        u: u_mat,
        s: Vec2::new(s1, s2),
        v: v_mat,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spirv_std::glam::{Mat2, Vec2};

    fn frob(m: Mat2) -> f32 {
        // Frobenius norm
        let a = m.to_cols_array(); // [m11, m21, m12, m22] (column-major)
        (a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3]).sqrt()
    }

    fn is_orthonormal(q: Mat2, tol: f32) -> bool {
        let i = Mat2::from_cols(Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0));
        let qtq = q.transpose() * q;
        frob(qtq - i) <= tol
    }

    fn det_mat2(m: Mat2) -> f32 {
        m.x_axis.x * m.y_axis.y - m.y_axis.x * m.x_axis.y
    }

    fn symmetric_offdiag_norm(m: Mat2) -> f32 {
        // For testing V^T (A^T A) V ≈ diagonal
        let off = m.x_axis.y.abs() + m.y_axis.x.abs();
        off
    }

    #[test]
    fn exact_reconstructs_known_cases() {
        let cases = [
            // diagonal, already sorted
            Mat2::from_cols(Vec2::new(3.0, 0.0), Vec2::new(0.0, 2.0)),
            // upper-triangular
            Mat2::from_cols(Vec2::new(3.0, 0.0), Vec2::new(1.0, 2.0)),
            // general
            Mat2::from_cols(Vec2::new(-1.0, 4.0), Vec2::new(2.0, -3.0)),
            // zero
            Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(0.0, 0.0)),
            // rank-1 (collinear columns)
            {
                let c0 = Vec2::new(2.5, -7.5);
                let c1 = c0 * 3.0;
                Mat2::from_cols(c0, c1)
            },
            // extreme scale separation (safe in f32)
            Mat2::from_cols(Vec2::new(1.0e8, 0.0), Vec2::new(0.0, 1.0e-4)),
            // negative determinant
            Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, -4.0)),
        ];

        for a in cases {
            let svd = svd2x2_exact(a);
            let sigma = Mat2::from_cols(Vec2::new(svd.s.x, 0.0), Vec2::new(0.0, svd.s.y));
            let a_hat = svd.u * sigma * svd.v.transpose();

            let tol = 1e-5 * (1.0 + frob(a));
            let recon_err = frob(a_hat - a);
            assert!(
                recon_err <= tol,
                "reconstruction error too large: frob(a_hat - a)={} tol={}, a={:?}",
                recon_err,
                tol,
                a
            );

            // Orthonormal U,V
            let u_qtq = svd.u.transpose() * svd.u;
            let u_err = frob(u_qtq - Mat2::from_cols(Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)));
            assert!(
                is_orthonormal(svd.u, 5.0e-6),
                "U not orthonormal: frob(U^T U - I)={}, U={:?}",
                u_err,
                svd.u
            );
            let v_qtq = svd.v.transpose() * svd.v;
            let v_err = frob(v_qtq - Mat2::from_cols(Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)));
            assert!(
                is_orthonormal(svd.v, 5.0e-6),
                "V not orthonormal: frob(V^T V - I)={}, V={:?}",
                v_err,
                svd.v
            );

            // Ordering and nonnegativity
            assert!(
                svd.s.x >= svd.s.y - 1e-6,
                "Singular values not ordered: s1={} s2={}, s={:?}",
                svd.s.x,
                svd.s.y,
                svd.s
            );
            assert!(
                svd.s.x >= 0.0 && svd.s.y >= 0.0,
                "Singular values not nonnegative: s1={} s2={}, s={:?}",
                svd.s.x,
                svd.s.y,
                svd.s
            );

            // Invariants:
            // 1) s1*s2 = |det(A)|  (exact in real arithmetic)
            let det_a = det_mat2(a).abs();
            let prod = svd.s.x * svd.s.y;
            assert!(
                (prod - det_a).abs() <= 5e-5 * (1.0 + det_a),
                "det invariant failed: (svd.s.x * svd.s.y)={} det_mat2(a).abs()={}",
                prod,
                det_a
            );

            // 2) s1^2 + s2^2 = ||A||_F^2
            let frob2 = frob(a).powi(2);
            let sumsq = svd.s.x * svd.s.x + svd.s.y * svd.s.y;
            assert!(
                (sumsq - frob2).abs() <= 5e-4 * (1.0 + frob2),
                "Frobenius invariant failed: sumsq(s)={} frob2={}",
                sumsq,
                frob2
            );

            // 3) V should diagonalize A^T A
            let ata = a.transpose() * a;
            let diag = svd.v.transpose() * ata * svd.v;
            let offdiag = symmetric_offdiag_norm(diag);
            let tol_diag = 1e-4 * (1.0 + frob(ata));
            assert!(
                offdiag <= tol_diag,
                "AtA not diagonalized: offdiag_norm={} tol={}, diag={:?}",
                offdiag,
                tol_diag,
                diag
            );
        }
    }

    #[test]
    fn exact_handles_negative_det_signs() {
        // If det(A) < 0 and Σ >= 0, then det(U)*det(V) must be -1
        let a = Mat2::from_cols(Vec2::new(2.0, 1.0), Vec2::new(3.0, -5.0)); // det < 0
        let svd = svd2x2_exact(a);
        let sign = (svd.u.determinant() * svd.v.determinant()).signum();
        let sign_a = det_mat2(a).signum();
        let sign_diff = (sign - sign_a).abs();
        assert!(
            sign_diff < 1e-6,
            "det(U)*det(V) should match sign(det(A)): det(U)*det(V)={} sign(det(A))={} diff={}",
            sign,
            sign_a,
            sign_diff
        );

        // Reconstruct exactly
        let sigma = Mat2::from_cols(Vec2::new(svd.s.x, 0.0), Vec2::new(0.0, svd.s.y));
        let a_hat = svd.u * sigma * svd.v.transpose();
        let recon_err = frob(a_hat - a);
        let tol = 1e-5 * (1.0 + frob(a));
        assert!(
            recon_err <= tol,
            "reconstruction error: frob(a_hat - a)={} tol={}",
            recon_err,
            tol
        );
    }

    #[test]
    fn exact_rank_deficient_behaviour() {
        // Perfect rank-1: second column is multiple of first
        let c0 = Vec2::new(-4.0, 1.0);
        let a = Mat2::from_cols(c0, c0 * 0.25);
        let svd = svd2x2_exact(a);

        let s2_tol = 1e-6 * (1.0 + svd.s.x);
        assert!(
            svd.s.y <= s2_tol,
            "s2 should be ~0 for rank-1: s2={} tol={} s1={}, SVD: {:?}",
            svd.s.y,
            s2_tol,
            svd.s.x,
            svd
        );
        // u2 should be perpendicular to u1
        let dot = svd.u.x_axis.dot(svd.u.y_axis).abs();
        assert!(
            dot <= 1e-5,
            "U columns should be perpendicular: |u0·u1|={} tol=1e-5",
            dot
        );
    }

    #[test]
    fn exact_identity_matrix() {
        // Identity matrix should have singular values (1, 1) and U = V = I
        let a = Mat2::from_cols(Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0));
        let svd = svd2x2_exact(a);

        let s1_err = (svd.s.x - 1.0).abs();
        assert!(
            s1_err <= 1e-6,
            "s1 should be 1.0: s1={} error={}",
            svd.s.x,
            s1_err
        );
        let s2_err = (svd.s.y - 1.0).abs();
        assert!(
            s2_err <= 1e-6,
            "s2 should be 1.0: s2={} error={}",
            svd.s.y,
            s2_err
        );

        // Reconstruction should be exact
        let sigma = Mat2::from_cols(Vec2::new(svd.s.x, 0.0), Vec2::new(0.0, svd.s.y));
        let a_hat = svd.u * sigma * svd.v.transpose();
        let recon_err = frob(a_hat - a);
        assert!(
            recon_err <= 1e-6,
            "reconstruction error: frob(a_hat - a)={} tol=1e-6",
            recon_err
        );
    }

    #[test]
    fn exact_nearly_singular_matrices() {
        // Test matrices that are nearly singular (small but non-zero determinant)
        let cases = [
            // Nearly collinear columns
            Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(1.001, 2.002)),
            // Small determinant with large entries
            Mat2::from_cols(Vec2::new(1000.0, 999.0), Vec2::new(1001.0, 1000.0)),
        ];

        for a in cases {
            let svd = svd2x2_exact(a);

            // Should satisfy all invariants even for nearly singular matrices
            let det_a = det_mat2(a).abs();
            let prod = svd.s.x * svd.s.y;
            let det_err = (prod - det_a).abs();
            let det_tol = 1e-3 * (1.0 + det_a);
            assert!(
                det_err <= det_tol,
                "det invariant failed for nearly singular matrix: |s1*s2 - |det(A)||={} tol={}, s1*s2={} |det(A)|={} a={:?}",
                det_err,
                det_tol,
                prod,
                det_a,
                a
            );

            // Reconstruction
            let sigma = Mat2::from_cols(Vec2::new(svd.s.x, 0.0), Vec2::new(0.0, svd.s.y));
            let a_hat = svd.u * sigma * svd.v.transpose();
            let recon_err = frob(a_hat - a);
            let recon_tol = 1e-4 * (1.0 + frob(a));
            assert!(
                recon_err <= recon_tol,
                "reconstruction error: frob(a_hat - a)={} tol={} a={:?}",
                recon_err,
                recon_tol,
                a
            );
        }
    }

    #[test]
    fn exact_rotation_matrices() {
        // Rotation matrices should have singular values (1, 1) and det = +1
        use core::f32::consts::PI;
        let angles = [0.0f32, PI / 4.0, PI / 2.0, PI];

        for theta in angles {
            let (s, c) = theta.sin_cos();
            let rot = Mat2::from_cols(Vec2::new(c, s), Vec2::new(-s, c));
            let svd = svd2x2_exact(rot);

            let s1_err = (svd.s.x - 1.0).abs();
            assert!(
                s1_err <= 1e-6,
                "s1 should be 1.0 for rotation: s1={} error={} theta={}",
                svd.s.x,
                s1_err,
                theta
            );
            let s2_err = (svd.s.y - 1.0).abs();
            assert!(
                s2_err <= 1e-6,
                "s2 should be 1.0 for rotation: s2={} error={} theta={}",
                svd.s.y,
                s2_err,
                theta
            );
            let det = det_mat2(rot);
            let det_err = (det - 1.0).abs();
            assert!(
                det_err <= 1e-6,
                "det should be 1.0 for rotation: det={} error={} theta={}",
                det,
                det_err,
                theta
            );
        }
    }

    // -------- Property-based testing over many random matrices --------
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn exact_properties_random(
            a11 in -1.0e3f32..1.0e3,
            a12 in -1.0e3f32..1.0e3,
            a21 in -1.0e3f32..1.0e3,
            a22 in -1.0e3f32..1.0e3,
        ) {
            let a = Mat2::from_cols(Vec2::new(a11, a21), Vec2::new(a12, a22));
            let svd = svd2x2_exact(a);

            let sigma = Mat2::from_cols(Vec2::new(svd.s.x, 0.0), Vec2::new(0.0, svd.s.y));
            let a_hat = svd.u * sigma * svd.v.transpose();

            let tol = 5e-5 * (1.0 + frob(a));
            let recon_err = frob(a_hat - a);
            prop_assert!(recon_err <= tol, "reconstruction error: frob(a_hat - a)={} tol={}", recon_err, tol);

            // Orthogonality and ordering
            prop_assert!(is_orthonormal(svd.u, 5e-5), "U not orthonormal: {:?}", svd.u);
            prop_assert!(is_orthonormal(svd.v, 5e-5), "V not orthonormal: {:?}", svd.v);
            prop_assert!(svd.s.x >= svd.s.y - 1e-5, "Singular values not ordered: s1={} s2={}, s={:?}", svd.s.x, svd.s.y, svd.s);
            prop_assert!(svd.s.x >= 0.0 && svd.s.y >= 0.0, "Singular values not nonnegative: s1={} s2={}, s={:?}", svd.s.x, svd.s.y, svd.s);

            // Invariants
            let det_a = det_mat2(a).abs();
            let prod = svd.s.x * svd.s.y;
            prop_assert!((prod - det_a).abs() <= 1e-3 * (1.0 + det_a), "Det invariant failed: (svd.s.x * svd.s.y)={} det_mat2(a).abs()={}", prod, det_a);

            let frob2 = frob(a).powi(2);
            let sumsq = svd.s.x * svd.s.x + svd.s.y * svd.s.y;
            prop_assert!((sumsq - frob2).abs() <= 1e-3 * (1.0 + frob2), "Frobenius invariant failed: sumsq(s)={} frob2={}", sumsq, frob2 );
        }
    }
}
