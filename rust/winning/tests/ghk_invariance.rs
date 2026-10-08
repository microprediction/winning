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
