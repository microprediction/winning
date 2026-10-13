//! The second-level (pair) cavity behind the top-k and rank Jacobian
//! off-diagonals takes the bounded two-direction rule (#511 residual):
//! the one-direction choice returned subtraction residue 1e5..1e75 times
//! the directly multiplied pair term on these fields.
use winning::{ndtr, rank_jacobian_kernel, top_k_jacobians_kernel};

fn direct_pair_jacobian(mu: &[f64], k: usize, lo: f64, hi: f64, points: usize,
                        rank: bool) -> Vec<f64> {
    let n = mu.len();
    let dx = (hi - lo) / (points - 1) as f64;
    let mut jac = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            if j == i {
                continue;
            }
            let mut s = 0.0;
            for l in 0..points {
                let x = lo + l as f64 * dx;
                let mut c = vec![0.0; n];
                c[0] = 1.0;
                for (r, &m) in mu.iter().enumerate() {
                    if r == i || r == j {
                        continue;
                    }
                    let (f, sv) = (ndtr(x - m), ndtr(m - x));
                    for t in (1..n).rev() {
                        c[t] = c[t] * sv + c[t - 1] * f;
                    }
                    c[0] *= sv;
                }
                let hi_c = if k <= n - 1 { c[k - 1] } else { 0.0 };
                let coef = if rank && k >= 2 { hi_c - c[k - 2] } else { hi_c };
                let d = |m: f64| (-0.5 * (x - m) * (x - m)).exp() / (2.0 * std::f64::consts::PI).sqrt();
                s += coef * d(mu[i]) * d(mu[j]);
            }
            jac[i * n + j] = s * dx;
        }
    }
    jac
}

#[test]
fn pair_cavities_match_a_direct_product() {
    let fields: [&[f64]; 3] = [&[-12.0, -4.0, 4.0, 12.0], &[-14.0, 0.0, 0.0],
                               &[-20.0, -6.0, 0.0, 3.0, 9.0]];
    for mu in fields {
        let n = mu.len();
        let sd = vec![1.0; n];
        let (lo, hi, points) = (mu[0] - 10.0, mu[n - 1] + 10.0, 513);
        for k in 1..n {
            let (jm, _) = top_k_jacobians_kernel(mu, &sd, k, lo, hi, points);
            let want = direct_pair_jacobian(mu, k, lo, hi, points, false);
            let (_, jr) = rank_jacobian_kernel(mu, &sd, k, lo, hi, points);
            let want_r = direct_pair_jacobian(mu, k, lo, hi, points, true);
            for idx in 0..n * n {
                if idx / n == idx % n {
                    continue;
                }
                if want[idx].abs() > 1e-280 {
                    let rel = (jm[idx] / want[idx] - 1.0).abs();
                    assert!(rel < 1e-9, "{mu:?} k={k} {idx}: {} vs {}", jm[idx], want[idx]);
                }
                // a rank entry is a difference: compare where it is not
                // itself cancellation at the scale of its terms
                if want_r[idx].abs() > 1e-280 && want_r[idx].abs() > 1e-12 * want[idx].abs() {
                    let rel = (jr[idx] / want_r[idx] - 1.0).abs();
                    assert!(rel < 1e-6, "rank {mu:?} r={k} {idx}: {} vs {}", jr[idx], want_r[idx]);
                }
            }
        }
    }
}
