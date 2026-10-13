//! GHK keeps its tail slope (#270): the per-draw factors and the
//! truncated draws used to be floored at 1e-300, so a far-trailing
//! alternative's estimate plateaued at 1e-300 (log -690.78).
use winning::{ghk_logprob_one_factor, log_ndtr, ndtri_log};

#[test]
fn binary_tail_is_exact_below_the_old_floor() {
    // one truncation step: every draw carries the same weight
    for &g in &[10.0, 53.0, 80.0, 200.0] {
        let lp = ghk_logprob_one_factor(&[0.0, g], &[], 0, &[1.0, 1.0], 2, 0, 16, 9);
        let exact = log_ndtr(-g / 2f64.sqrt());
        assert!((lp - exact).abs() < 1e-9 * exact.abs(), "g = {g}: {lp} vs {exact}");
    }
}

#[test]
fn three_way_tail_tracks_an_independent_quadrature() {
    // references: 1-d conditional-normal quadrature in log space
    let v = [0.3, -0.2, 0.1];
    for &(g, reference) in &[(-50.0, -789.4427401866274), (-60.0, -1131.5948698859718),
                             (-80.0, -2001.2502944154676)] {
        let lp = ghk_logprob_one_factor(&[g, 0.0, 0.5], &v, 1, &[1.0, 1.0, 1.0],
                                        3, 0, 4000, 9);
        assert!((lp - reference).abs() < 0.05, "g = {g}: {lp} vs {reference}");
    }
}

#[test]
fn ndtri_log_inverts_log_ndtr_in_the_deep_tail() {
    for &x in &[-5.0, -30.0, -38.0, -40.0, -60.0, -200.0] {
        let back = ndtri_log(log_ndtr(x));
        assert!((back - x).abs() < 1e-9 * x.abs(), "x = {x}: {back}");
    }
}
