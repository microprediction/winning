//! fastrace: pyo3 frontend over fastrace-core (the pure-rust kernels).
//! All numerics live in fastrace-core; this crate only converts numpy
//! arrays and releases the GIL. The future R frontend (extendr) wraps
//! the same core.

use winning::*;
use winning::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;


/// Min-wins factor-race win probabilities (normalized), raw own-location
/// slopes of the unnormalized map (the inversion preconditioner), and the
/// pre-normalization total. slope_i = d p_raw_i / d mu_i.
#[pyfunction]
#[pyo3(signature = (mu, v, d, f, w, points=501, lo=f64::NAN, hi=f64::NAN))]
fn forward_and_slopes<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray1<f64>,
    v: PyReadonlyArray2<f64>,
    d: PyReadonlyArray1<f64>,
    f: PyReadonlyArray2<f64>,
    w: PyReadonlyArray1<f64>,
    points: usize,
    lo: f64,
    hi: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, f64)> {
    let mu_o: Array1<f64> = mu.as_array().to_owned();
    let v_o: Array2<f64> = v.as_array().to_owned();
    let d_o: Array1<f64> = d.as_array().to_owned();
    let f_o: Array2<f64> = f.as_array().to_owned();
    let w_o: Array1<f64> = w.as_array().to_owned();
    let (p, sl, total) = py.allow_threads(|| {
        forward_kernel(mu_o.view(), v_o.view(), d_o.view(), f_o.view(),
                       w_o.view(), points, lo, hi)
    });
    Ok((p.into_pyarray_bound(py), sl.into_pyarray_bound(py), total))
}


/// Back-compatible forward-only entry point: (normalized p, total).
#[pyfunction]
#[pyo3(signature = (mu, v, d, f, w, points=501, lo=f64::NAN, hi=f64::NAN))]
fn win_probabilities_factor<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray1<f64>,
    v: PyReadonlyArray2<f64>,
    d: PyReadonlyArray1<f64>,
    f: PyReadonlyArray2<f64>,
    w: PyReadonlyArray1<f64>,
    points: usize,
    lo: f64,
    hi: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let mu_o: Array1<f64> = mu.as_array().to_owned();
    let v_o: Array2<f64> = v.as_array().to_owned();
    let d_o: Array1<f64> = d.as_array().to_owned();
    let f_o: Array2<f64> = f.as_array().to_owned();
    let w_o: Array1<f64> = w.as_array().to_owned();
    let (p, _sl, total) = py.allow_threads(|| {
        forward_kernel(mu_o.view(), v_o.view(), d_o.view(), f_o.view(),
                       w_o.view(), points, lo, hi)
    });
    Ok((p.into_pyarray_bound(py), total))
}


/// Jacobian-vector product of the min-wins map. form "ibp" is the
/// continuum weighted-Laplacian derivative; form "grid" is the exact
/// derivative of the frozen-grid rectangle sum. Mirrors
/// raceutil.jacobian_vector_product.
#[pyfunction]
#[pyo3(signature = (mu, v, d, f, w, h, points=3001, form="ibp"))]
#[allow(clippy::too_many_arguments)]
fn jacobian_vector_product<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray1<f64>,
    v: PyReadonlyArray2<f64>,
    d: PyReadonlyArray1<f64>,
    f: PyReadonlyArray2<f64>,
    w: PyReadonlyArray1<f64>,
    h: PyReadonlyArray1<f64>,
    points: usize,
    form: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mu_o: Array1<f64> = mu.as_array().to_owned();
    let v_o: Array2<f64> = v.as_array().to_owned();
    let d_o: Array1<f64> = d.as_array().to_owned();
    let f_o: Array2<f64> = f.as_array().to_owned();
    let w_o: Array1<f64> = w.as_array().to_owned();
    let h_o: Array1<f64> = h.as_array().to_owned();
    let grid = form == "grid";
    let out = py.allow_threads(|| {
        jvp_kernel(mu_o.view(), v_o.view(), d_o.view(), f_o.view(),
                   w_o.view(), h_o.view(), points, grid)
    });
    Ok(out.into_pyarray_bound(py))
}


/// Chebyshev-separated forward pass: O(Q r (N + L)) per the exp20
/// prototype, with exponential convergence in (rm, rs). Returns
/// (normalized shares, pre-normalization total).
#[pyfunction]
#[pyo3(signature = (mu, v, d, f, w, points=1501, rm=48, rs=14))]
#[allow(clippy::too_many_arguments)]
fn win_probabilities_factor_separated<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray1<f64>,
    v: PyReadonlyArray2<f64>,
    d: PyReadonlyArray1<f64>,
    f: PyReadonlyArray2<f64>,
    w: PyReadonlyArray1<f64>,
    points: usize,
    rm: usize,
    rs: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let mu_o: Array1<f64> = mu.as_array().to_owned();
    let v_o: Array2<f64> = v.as_array().to_owned();
    let d_o: Array1<f64> = d.as_array().to_owned();
    let f_o: Array2<f64> = f.as_array().to_owned();
    let w_o: Array1<f64> = w.as_array().to_owned();
    let (p, total) = py.allow_threads(|| {
        separated_kernel(mu_o.view(), v_o.view(), d_o.view(), f_o.view(),
                         w_o.view(), points, rm, rs)
    });
    Ok((p.into_pyarray_bound(py), total))
}


/// All-alternative GHK shares (per-alternative sequential importance
/// sampling), parallel over alternatives, normalized by the sum. The
/// like-for-like compiled version of the Python baseline.
#[pyfunction]
#[pyo3(signature = (mu, v, d, r_draws=1000, seed=9))]
fn ghk_all_shares<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray1<f64>,
    v: PyReadonlyArray2<f64>,
    d: PyReadonlyArray1<f64>,
    r_draws: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mu_o: Array1<f64> = mu.as_array().to_owned();
    let v_o: Array2<f64> = v.as_array().to_owned();
    let d_o: Array1<f64> = d.as_array().to_owned();
    let out = py.allow_threads(|| {
        let n = mu_o.len();
        let k = v_o.ncols();
        let mut sigma = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sv = 0.0;
                for r_ in 0..k {
                    sv += v_o[[i, r_]] * v_o[[j, r_]];
                }
                sigma[i * n + j] = sv + if i == j { d_o[i] } else { 0.0 };
            }
        }
        let mus: Vec<f64> = mu_o.iter().cloned().collect();
        let p: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| ghk_prob_one(&mus, &sigma, n, i, r_draws,
                                  seed.wrapping_add(i as u64)))
            .collect();
        let total: f64 = p.iter().sum();
        Array1::from_iter(p.into_iter().map(|x| x / total))
    });
    Ok(out.into_pyarray_bound(py))
}


#[pyfunction]
#[pyo3(signature = (mu, sd, v, starts, a_nodes, a_weights, points=257, lo=f64::NAN, hi=f64::NAN, fast_max_entries=10_000_000))]
fn block_race<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray1<f64>,
    sd: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
    starts: PyReadonlyArray1<i64>,
    a_nodes: PyReadonlyArray1<f64>,
    a_weights: PyReadonlyArray1<f64>,
    points: usize,
    lo: f64,
    hi: f64,
    fast_max_entries: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mu_o: Array1<f64> = mu.as_array().to_owned();
    let sd_o: Array1<f64> = sd.as_array().to_owned();
    let v_o: Array1<f64> = v.as_array().to_owned();
    let st: Vec<usize> = starts.as_array().iter().map(|&x| x as usize).collect();
    let an: Array1<f64> = a_nodes.as_array().to_owned();
    let aw: Array1<f64> = a_weights.as_array().to_owned();
    let p = py.allow_threads(|| {
        block_kernel(mu_o.view(), sd_o.view(), v_o.view(), &st, an.view(),
                     aw.view(), points, lo, hi, fast_max_entries)
    });
    Ok(p.into_pyarray_bound(py))
}


#[pyfunction]
#[pyo3(signature = (mu, sd, v, starts, nodes, weights, points=257, lo=f64::NAN, hi=f64::NAN))]
fn block_race_r<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray1<f64>,
    sd: PyReadonlyArray1<f64>,
    v: PyReadonlyArray2<f64>,
    starts: PyReadonlyArray1<i64>,
    nodes: PyReadonlyArray2<f64>,
    weights: PyReadonlyArray1<f64>,
    points: usize,
    lo: f64,
    hi: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mu_o: Array1<f64> = mu.as_array().to_owned();
    let sd_o: Array1<f64> = sd.as_array().to_owned();
    let v_o: Array2<f64> = v.as_array().to_owned();
    let st: Vec<usize> = starts.as_array().iter().map(|&x| x as usize).collect();
    let nd: Array2<f64> = nodes.as_array().to_owned();
    let ww: Array1<f64> = weights.as_array().to_owned();
    let p = py.allow_threads(|| {
        block_kernel_r(mu_o.view(), sd_o.view(), v_o.view(), &st, nd.view(),
                       ww.view(), points, lo, hi)
    });
    Ok(p.into_pyarray_bound(py))
}


#[pyfunction]
#[pyo3(signature = (mu, sd, v, starts, parent, lam, a_nodes, a_weights, points, lo, hi))]
#[allow(clippy::too_many_arguments)]
fn tree_race<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray1<f64>,
    sd: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
    starts: PyReadonlyArray1<i64>,
    parent: PyReadonlyArray1<i64>,
    lam: PyReadonlyArray1<f64>,
    a_nodes: PyReadonlyArray1<f64>,
    a_weights: PyReadonlyArray1<f64>,
    points: usize,
    lo: f64,
    hi: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mu_o: Vec<f64> = mu.as_array().to_vec();
    let sd_o: Vec<f64> = sd.as_array().to_vec();
    let v_o: Vec<f64> = v.as_array().to_vec();
    let st: Vec<usize> = starts.as_array().iter().map(|&x| x as usize).collect();
    let pa: Vec<i64> = parent.as_array().to_vec();
    let lm: Vec<f64> = lam.as_array().to_vec();
    let an: Vec<f64> = a_nodes.as_array().to_vec();
    let aw: Vec<f64> = a_weights.as_array().to_vec();
    let p = py.allow_threads(|| {
        tree_kernel(&mu_o, &sd_o, &v_o, &st, &pa, &lm, &an, &aw, points, lo, hi)
    });
    Ok(Array1::from_vec(p).into_pyarray_bound(py))
}


/// state_prices_from_offsets: field from the shifted base density, then
/// implicit prices AT those offsets (unnormalized, as the reference).
#[pyfunction]
fn classic_state_prices(density: Vec<f64>, offsets: Vec<f64>) -> PyResult<Vec<f64>> {
    let l = ((density.len() - 1) / 2) as i64;
    let base_cdf = pdf_to_cdf(&density);
    let cdfs: Vec<Vec<f64>> = offsets.iter().map(|&o| shifted_cdf(&base_cdf, o, l)).collect();
    let (cdf_all, mult_all) = winner_of_many(&cdfs);
    Ok(implicit_prices(&base_cdf, &cdf_all, &mult_all, &offsets, l))
}


/// solve_for_implied_offsets: the reference fixed-point iteration --
/// interpolation table offset -> price rebuilt against the current field.
#[pyfunction]
#[pyo3(signature = (density, prices, offset_samples, guess, n_iter=3))]
fn classic_calibrate(
    density: Vec<f64>,
    prices: Vec<f64>,
    offset_samples: Vec<f64>,
    guess: Vec<f64>,
    n_iter: usize,
) -> PyResult<Vec<f64>> {
    let l = ((density.len() - 1) / 2) as i64;
    let base_cdf = pdf_to_cdf(&density);
    let mut cdfs: Vec<Vec<f64>> =
        guess.iter().map(|&o| shifted_cdf(&base_cdf, o, l)).collect();
    // offset_samples arrive descending (better first); implied prices are
    // then ascending, which is what the interpolation needs
    let mut implied: Vec<f64> = prices.clone();
    for _ in 0..n_iter {
        let (cdf_all, mult_all) = winner_of_many(&cdfs);
        let table = implicit_prices(&base_cdf, &cdf_all, &mult_all, &offset_samples, l);
        implied = prices
            .iter()
            .map(|&p| interp1(p, &table, &offset_samples))
            .collect();
        cdfs = implied.iter().map(|&o| shifted_cdf(&base_cdf, o, l)).collect();
    }
    Ok(implied)
}


#[pymodule]
fn fastrace(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(forward_and_slopes, m)?)?;
    m.add_function(wrap_pyfunction!(block_race, m)?)?;
    m.add_function(wrap_pyfunction!(block_race_r, m)?)?;
    m.add_function(wrap_pyfunction!(jacobian_vector_product, m)?)?;
    m.add_function(wrap_pyfunction!(win_probabilities_factor_separated, m)?)?;
    m.add_function(wrap_pyfunction!(ghk_all_shares, m)?)?;
    m.add_function(wrap_pyfunction!(win_probabilities_factor, m)?)?;
    m.add_function(wrap_pyfunction!(tree_race, m)?)?;
    m.add_function(wrap_pyfunction!(top_k, m)?)?;
    m.add_function(wrap_pyfunction!(classic_state_prices, m)?)?;
    m.add_function(wrap_pyfunction!(classic_calibrate, m)?)?;
    Ok(())
}


/// Top-k membership probabilities, normal base; see winning::top_k_kernel.
#[pyfunction]
fn top_k<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray1<f64>,
    sd: PyReadonlyArray1<f64>,
    k: usize,
    lo: f64,
    hi: f64,
    points: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let m: Vec<f64> = mu.as_array().to_vec();
    let s: Vec<f64> = sd.as_array().to_vec();
    let q = py.allow_threads(|| winning::top_k_kernel(&m, &s, k, lo, hi, points));
    Ok(Array1::from_vec(q).into_pyarray_bound(py))
}
