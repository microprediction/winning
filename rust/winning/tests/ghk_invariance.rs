//! GHK is homogeneous in the utility unit and blind to a common shock.
//! A binary race has one truncation step, so every draw carries the same
//! weight and these are exact checks, not Monte Carlo comparisons.
use winning::{ghk_prob_one, ghk_prob_one_factor, ndtr};

const EXACT: f64 = 0.760_249_938_906_523_3; // Phi(1/sqrt 2)

#[test]
fn scale_does_not_move_a_binary_race() {
    // #409, #101: the absolute 1e-12 ridge gave 0.7181 at c = 1e-6 and
    // 0.5040 at c = 1e-8
    for &c in &[1.0, 1e-6, 1e-8, 1e-12] {
        let mu = [0.0, c];
        let d = [c * c, c * c];
        let p1 = ghk_prob_one_factor(&mu, &[], 0, &d, 2, 1, 32, 9);
        assert!((p1 - EXACT).abs() < 1e-12, "c = {c}: {p1}");
        let sigma = [c * c, 0.0, 0.0, c * c];
        let p2 = ghk_prob_one(&mu, &sigma, 2, 1, 32, 9);
        assert!((p2 - EXACT).abs() < 1e-12, "dense, c = {c}: {p2}");
    }
}

#[test]
fn a_common_loading_cancels() {
    // #302: V = 1e8 on both rows materialised Sigma = 1e16 + 1 == 1e16
    // and priced the 76/24 race as [1e-300, 1]
    let mu = [0.0, 1.0];
    let d = [1.0, 1.0];
    for &v in &[0.0, 1e4, 1e8, 1e12] {
        let p = ghk_prob_one_factor(&mu, &[v, v], 1, &d, 2, 1, 8, 9);
        assert!((p - EXACT).abs() < 1e-12, "common loading {v}: {p}");
    }
    assert!((ndtr(1.0 / 2f64.sqrt()) - EXACT).abs() < 1e-15);
}

#[test]
fn subnormal_scales_are_still_the_same_race() {
    // #409: the relative ridge kept a 1e-300 floor, an absolute 1e-312
    // below it: D = 1e-310 gave 0.7597, D = 1e-320 gave 0.5000
    for &c in &[1e-150, 1e-155] {
        let mu = [0.0, c];
        let d = [c * c, c * c];
        let p = ghk_prob_one_factor(&mu, &[], 0, &d, 2, 1, 32, 9);
        assert!((p - EXACT).abs() < 1e-12, "c = {c}: {p}");
    }
}

#[test]
fn a_valid_small_pivot_is_not_inflated() {
    // #409: two near-twins share a unit factor with variance 2^-43 each,
    // their gap scaled to that direction; a third runner is independent.
    // The always-on relative ridge priced [0.5, 0.167, 0.333]; the exact
    // shares are [0.49999992, 0.07932763, 0.42067245].
    let delta = 2f64.powi(-43);
    let a = 0.5 * (2.0 * delta).sqrt();
    let mu = [0.0, -a, a];
    let v = [0.0, 1.0, 1.0];
    let d = [1.0, delta, delta];
    let p: Vec<f64> = (0..3)
        .map(|i| ghk_prob_one_factor(&mu, &v, 1, &d, 3, i, 200_000, 7 + i as u64))
        .collect();
    let s: f64 = p.iter().sum();
    let want = [0.49999992, 0.07932763, 0.42067245];
    for i in 0..3 {
        assert!((p[i] / s - want[i]).abs() < 3e-3, "share {i}: {}", p[i] / s);
    }
}

#[test]
fn an_overflowing_variance_sum_is_the_same_race() {
    // #409: D = 5e307 gives contrast variances 1e308 whose SUM overflowed;
    // the fallback put an infinite ridge on the diagonal, every pivot was
    // inf and the asymmetric race came back [1/3, 1/3, 1/3]. The same
    // race in units of 1e154 is the reference, at the same seeds.
    let big: Vec<f64> = (0..3)
        .map(|i| ghk_prob_one_factor(&[0.0, 1e154, -1e154], &[], 0, &[5e307; 3],
                                     3, i, 4096, 9 + i as u64))
        .collect();
    let unit: Vec<f64> = (0..3)
        .map(|i| ghk_prob_one_factor(&[0.0, 1.0, -1.0], &[], 0, &[0.5; 3],
                                     3, i, 4096, 9 + i as u64))
        .collect();
    for i in 0..3 {
        assert!(big[i].is_finite() && (big[i] - unit[i]).abs() < 1e-12,
                "share {i}: {} vs {}", big[i], unit[i]);
    }
    let sigma = [5e307, 0.0, 0.0, 0.0, 5e307, 0.0, 0.0, 0.0, 5e307];
    let dense = ghk_prob_one(&[0.0, 1e154, -1e154], &sigma, 3, 1, 4096, 10);
    assert!((dense - unit[1]).abs() < 1e-12, "dense: {dense} vs {}", unit[1]);
}

#[test]
fn an_ordinary_race_is_unchanged() {
    // values from main before the overflow-safe trace (9b29ddef)
    let want = [0.30292768905168094, 0.4424900848341717, 0.2544313709921934];
    for (i, w) in want.iter().enumerate() {
        let p = ghk_prob_one_factor(&[0.0, 0.3, -0.2], &[0.3, -0.1, 0.4], 1,
                                    &[1.0, 0.8, 1.2], 3, i, 500, 4);
        assert!((p - w).abs() < 1e-13 * w, "share {i}: {p} vs {w}");
    }
}
