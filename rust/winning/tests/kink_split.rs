//! The opt-in kink-split Laplace kernel (#541): the uniform lattice is
//! only second order across the density's kinks, so the direct Laplace
//! kernel missed the certified race by ~1e-4 at 257 points. The split
//! lattice (python's kink_lattice) reaches the window's own truncation.
//! The default kernel is unchanged.
use winning::ndarray::{array, Array2};
use winning::{forward_kernel_base, forward_kernel_base_split, gauss_legendre,
              kink_lattice, BaseSpec};

// winning.certified.certified_race_probabilities(mu, D, base="laplace")
const SURE: [f64; 4] = [0.6657629820244252, 0.07436987989771356,
                        0.057813301389397775, 0.2020538366884635];

#[test]
fn split_laplace_meets_the_certified_race() {
    let mu = array![-1.0, 0.3, 0.8, -0.2];
    let d = array![1.0, 0.7, 1.3, 0.9];
    let v = Array2::<f64>::zeros((4, 1));
    let f = Array2::<f64>::zeros((1, 1));
    let w = array![1.0];
    let (lo, hi) = (-23.409897328177294, 6.5036512849238814);
    let base = BaseSpec { id: 3, params: [0.0; 6] };
    let (pu, _, _) = forward_kernel_base(mu.view(), v.view(), d.view(),
                                         f.view(), w.view(), 257, lo, hi, base);
    let (ps, _, _) = forward_kernel_base_split(mu.view(), v.view(), d.view(),
                                               f.view(), w.view(), 257, lo, hi,
                                               base, &[0.0]);
    let err_u = (0..4).map(|i| (pu[i] - SURE[i]).abs()).fold(0.0, f64::max);
    let err_s = (0..4).map(|i| (ps[i] - SURE[i]).abs()).fold(0.0, f64::max);
    assert!(err_u > 1e-6, "uniform lattice error {err_u}");
    assert!(err_s < 1e-12, "split lattice error {err_s}");
}

#[test]
fn gauss_legendre_integrates_polynomials_exactly() {
    for m in [4usize, 5, 9, 33, 128] {
        let (t, w) = gauss_legendre(m);
        assert!(t.windows(2).all(|p| p[0] < p[1]));
        for deg in 0..(2 * m) {
            let q: f64 = t.iter().zip(&w).map(|(x, wi)| wi * x.powi(deg as i32)).sum();
            let exact = if deg % 2 == 1 { 0.0 } else { 2.0 / (deg as f64 + 1.0) };
            assert!((q - exact).abs() < 1e-13, "m {m} deg {deg}: {q} vs {exact}");
        }
    }
}

#[test]
fn kink_lattice_keeps_its_node_bound() {
    // a large field merges its kinks onto points / 4 bins: the node count
    // stays under about 2 * points however many runners there are
    let n = 5000;
    let m: Vec<f64> = (0..n).map(|i| -10.0 + 20.0 * i as f64 / n as f64).collect();
    let sd = vec![1.0; n];
    let (x, w) = kink_lattice(-20.0, 20.0, 257, &m, &sd, &[0.0]);
    assert!(x.len() <= 2 * 257 + 8, "{} nodes", x.len());
    assert!((w.iter().sum::<f64>() - 40.0).abs() < 1e-12);
    assert!(x.windows(2).all(|p| p[0] < p[1]));
}
