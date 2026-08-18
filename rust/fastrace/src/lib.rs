//! fastrace: fused kernel for the factor-probit share transform.
//!
//! Computes the min-wins factor race of the `thurstone`/`raceutil` line:
//!   p_i = E_f [ integral g_i(x|f) * prod_{j!=i} S_j(x|f) dx ],
//! with S the Gaussian survival, by the shared-survival-field identity
//! (build the field product once per node, divide one out per alternative).
//!
//! Design notes.
//! - Parallel over factor nodes (rayon): nodes are independent.
//! - x-tiled: per node, lattice columns are processed in tiles so the
//!   per-tile logS/logg blocks (N x TILE) stay in cache; two passes per
//!   tile (accumulate field, then distribute) instead of materializing
//!   N x L temporaries as NumPy must.
//! - log-domain throughout: log_ndtr directly (never 1 - Phi), analytic
//!   log-density. Mirrors the tail-stability fix in the Python reference.
//! - Not yet implemented (stacks on top): the FFT cross-correlation form
//!   of the per-node location->probability curves, which would let
//!   calibration Newton steps sample frozen curves by interpolation
//!   instead of re-integrating (the `winning` interpolation trick,
//!   factor generalization).

use numpy::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

const TILE: usize = 256;
const LN_SQRT_2PI: f64 = 0.918938533204672741780329736406;

/// log(Phi(z)). libm::erfc is exact to double precision until it
/// underflows (z < about -37.5); beyond that use the asymptotic series
/// log Phi(z) = -z^2/2 - log(-z) - log sqrt(2 pi) + log(1 - 1/z^2 + 3/z^4 - 15/z^6).
fn log_ndtr(z: f64) -> f64 {
    if z > 6.0 {
        -0.5 * libm::erfc(z * std::f64::consts::FRAC_1_SQRT_2)
    } else if z > -37.0 {
        (0.5 * libm::erfc(-z * std::f64::consts::FRAC_1_SQRT_2)).ln()
    } else {
        let z2 = z * z;
        -0.5 * z2 - LN_SQRT_2PI - (-z).ln()
            + (1.0 - 1.0 / z2 + 3.0 / (z2 * z2) - 15.0 / (z2 * z2 * z2)).ln()
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_kernel(
    mu: ArrayView1<f64>,
    v: ArrayView2<f64>,
    d: ArrayView1<f64>,
    f_nodes: ArrayView2<f64>,
    w: ArrayView1<f64>,
    points: usize,
) -> (Array1<f64>, Array1<f64>, f64) {
    let n = mu.len();
    let q = f_nodes.nrows();
    let sd: Vec<f64> = d.iter().map(|x| x.sqrt()).collect();
    let sd_max = sd.iter().cloned().fold(f64::MIN, f64::max);
    let log_norm: Vec<f64> = sd.iter().map(|s| s.ln() + LN_SQRT_2PI).collect();

    // conditional locations m[qi*n + i] and the global interval
    let mut lo = f64::MAX;
    let mut hi = f64::MIN;
    let mut m_all = vec![0.0f64; q * n];
    for qi in 0..q {
        for i in 0..n {
            let mut mi = mu[i];
            for r in 0..v.ncols() {
                mi += v[[i, r]] * f_nodes[[qi, r]];
            }
            m_all[qi * n + i] = mi;
            lo = lo.min(mi);
            hi = hi.max(mi);
        }
    }
    lo -= 8.0 * sd_max;
    hi += 8.0 * sd_max;
    let dx = (hi - lo) / (points - 1) as f64;

    let p: Vec<f64> = (0..q)
        .into_par_iter()
        .map(|qi| {
            let m = &m_all[qi * n..(qi + 1) * n];
            let wq = w[qi];
            let mut acc = vec![0.0f64; 2 * n];
            let mut logs = vec![0.0f64; n * TILE];
            let mut logg = vec![0.0f64; n * TILE];
            let mut field = vec![0.0f64; TILE];
            let mut t0 = 0;
            while t0 < points {
                let tl = TILE.min(points - t0);
                field[..tl].fill(0.0);
                for i in 0..n {
                    let inv_sd = 1.0 / sd[i];
                    let ln_i = log_norm[i];
                    let mi = m[i];
                    let row_s = &mut logs[i * TILE..i * TILE + tl];
                    let row_g = &mut logg[i * TILE..i * TILE + tl];
                    for t in 0..tl {
                        let x = lo + (t0 + t) as f64 * dx;
                        let z = (x - mi) * inv_sd;
                        let ls = log_ndtr(-z);
                        row_s[t] = ls;
                        row_g[t] = -0.5 * z * z - ln_i;
                        field[t] += ls;
                    }
                }
                for i in 0..n {
                    let row_s = &logs[i * TILE..i * TILE + tl];
                    let row_g = &logg[i * TILE..i * TILE + tl];
                    let inv_sd = 1.0 / sd[i];
                    let mi = m[i];
                    let mut s = 0.0f64;
                    let mut sl = 0.0f64;
                    for t in 0..tl {
                        let e = row_g[t] + field[t] - row_s[t];
                        if e > -745.0 {
                            let v = e.exp();
                            s += v;
                            let x = lo + (t0 + t) as f64 * dx;
                            sl += (x - mi) * inv_sd * inv_sd * v;
                        }
                    }
                    acc[i] += wq * s * dx;
                    acc[n + i] += wq * sl * dx;
                }
                t0 += tl;
            }
            acc
        })
        .reduce(
            || vec![0.0f64; 2 * n],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        );

    let total: f64 = p[..n].iter().sum();
    let p_norm: Array1<f64> = Array1::from_iter(p[..n].iter().map(|x| x / total));
    let slopes: Array1<f64> = Array1::from_iter(p[n..].iter().cloned());
    (p_norm, slopes, total)
}

/// Min-wins factor-race win probabilities (normalized), raw own-location
/// slopes of the unnormalized map (the inversion preconditioner), and the
/// pre-normalization total. slope_i = d p_raw_i / d mu_i.
#[pyfunction]
#[pyo3(signature = (mu, v, d, f, w, points=501))]
fn forward_and_slopes<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray1<f64>,
    v: PyReadonlyArray2<f64>,
    d: PyReadonlyArray1<f64>,
    f: PyReadonlyArray2<f64>,
    w: PyReadonlyArray1<f64>,
    points: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, f64)> {
    let mu_o: Array1<f64> = mu.as_array().to_owned();
    let v_o: Array2<f64> = v.as_array().to_owned();
    let d_o: Array1<f64> = d.as_array().to_owned();
    let f_o: Array2<f64> = f.as_array().to_owned();
    let w_o: Array1<f64> = w.as_array().to_owned();
    let (p, sl, total) = py.allow_threads(|| {
        forward_kernel(mu_o.view(), v_o.view(), d_o.view(), f_o.view(),
                       w_o.view(), points)
    });
    Ok((p.into_pyarray_bound(py), sl.into_pyarray_bound(py), total))
}

/// Back-compatible forward-only entry point: (normalized p, total).
#[pyfunction]
#[pyo3(signature = (mu, v, d, f, w, points=501))]
fn win_probabilities_factor<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray1<f64>,
    v: PyReadonlyArray2<f64>,
    d: PyReadonlyArray1<f64>,
    f: PyReadonlyArray2<f64>,
    w: PyReadonlyArray1<f64>,
    points: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let mu_o: Array1<f64> = mu.as_array().to_owned();
    let v_o: Array2<f64> = v.as_array().to_owned();
    let d_o: Array1<f64> = d.as_array().to_owned();
    let f_o: Array2<f64> = f.as_array().to_owned();
    let w_o: Array1<f64> = w.as_array().to_owned();
    let (p, _sl, total) = py.allow_threads(|| {
        forward_kernel(mu_o.view(), v_o.view(), d_o.view(), f_o.view(),
                       w_o.view(), points)
    });
    Ok((p.into_pyarray_bound(py), total))
}


#[allow(clippy::too_many_arguments)]
fn jvp_kernel(
    mu: ArrayView1<f64>,
    v: ArrayView2<f64>,
    d: ArrayView1<f64>,
    f_nodes: ArrayView2<f64>,
    w: ArrayView1<f64>,
    h: ArrayView1<f64>,
    points: usize,
    grid_form: bool,
) -> Array1<f64> {
    let n = mu.len();
    let q = f_nodes.nrows();
    let sd: Vec<f64> = d.iter().map(|x| x.sqrt()).collect();
    let sd_max = sd.iter().cloned().fold(f64::MIN, f64::max);
    let log_norm: Vec<f64> = sd.iter().map(|s| s.ln() + LN_SQRT_2PI).collect();
    let hvec: Vec<f64> = h.iter().cloned().collect();

    let mut lo = f64::MAX;
    let mut hi = f64::MIN;
    let mut m_all = vec![0.0f64; q * n];
    for qi in 0..q {
        for i in 0..n {
            let mut mi = mu[i];
            for r in 0..v.ncols() {
                mi += v[[i, r]] * f_nodes[[qi, r]];
            }
            m_all[qi * n + i] = mi;
            lo = lo.min(mi);
            hi = hi.max(mi);
        }
    }
    lo -= 8.0 * sd_max;
    hi += 8.0 * sd_max;
    let dx = (hi - lo) / (points - 1) as f64;

    let out: Vec<f64> = (0..q)
        .into_par_iter()
        .map(|qi| {
            let m = &m_all[qi * n..(qi + 1) * n];
            let wq = w[qi];
            let mut acc = vec![0.0f64; n];
            let mut logs = vec![0.0f64; n * TILE];
            let mut logg = vec![0.0f64; n * TILE];
            let mut haz = vec![0.0f64; n * TILE];
            let mut field = vec![0.0f64; TILE];
            let mut asum = vec![0.0f64; TILE];   // A = sum_j h_j haz_j
            let mut lsum = vec![0.0f64; TILE];   // Lambda = sum_j haz_j
            let mut t0 = 0;
            while t0 < points {
                let tl = TILE.min(points - t0);
                field[..tl].fill(0.0);
                asum[..tl].fill(0.0);
                lsum[..tl].fill(0.0);
                for i in 0..n {
                    let inv_sd = 1.0 / sd[i];
                    let ln_i = log_norm[i];
                    let mi = m[i];
                    let hi_ = hvec[i];
                    for t in 0..tl {
                        let x = lo + (t0 + t) as f64 * dx;
                        let z = (x - mi) * inv_sd;
                        let ls = log_ndtr(-z);
                        let lg = -0.5 * z * z - ln_i;
                        let hz = (lg - ls).exp();
                        logs[i * TILE + t] = ls;
                        logg[i * TILE + t] = lg;
                        haz[i * TILE + t] = hz;
                        field[t] += ls;
                        asum[t] += hi_ * hz;
                        lsum[t] += hz;
                    }
                }
                for i in 0..n {
                    let hi_ = hvec[i];
                    let mi = m[i];
                    let inv_d = 1.0 / d[i];
                    let mut s = 0.0f64;
                    for t in 0..tl {
                        let e = logg[i * TILE + t] + field[t] - logs[i * TILE + t];
                        if e > -745.0 {
                            let g_r = e.exp();
                            let term = if grid_form {
                                let x = lo + (t0 + t) as f64 * dx;
                                hi_ * (x - mi) * inv_d + asum[t]
                                    - hi_ * haz[i * TILE + t]
                            } else {
                                asum[t] - hi_ * lsum[t]
                            };
                            s += g_r * term;
                        }
                    }
                    acc[i] += wq * s * dx;
                }
                t0 += tl;
            }
            acc
        })
        .reduce(
            || vec![0.0f64; n],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        );
    Array1::from_vec(out)
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


fn cheb_nodes(a: f64, b: f64, r: usize) -> Vec<f64> {
    (0..r)
        .map(|k| 0.5 * (a + b)
            + 0.5 * (b - a)
                * ((2 * k + 1) as f64 * std::f64::consts::PI / (2 * r) as f64).cos())
        .collect()
}

fn bary_weights(nodes: &[f64]) -> Vec<f64> {
    let r = nodes.len();
    (0..r)
        .map(|j| {
            let mut w = 1.0;
            for k in 0..r {
                if k != j {
                    w /= nodes[j] - nodes[k];
                }
            }
            w
        })
        .collect()
}

/// Barycentric Lagrange interpolation row for query point q.
fn bary_row(nodes: &[f64], wts: &[f64], q: f64) -> Vec<f64> {
    let r = nodes.len();
    for j in 0..r {
        if (q - nodes[j]).abs() < 1e-14 {
            let mut row = vec![0.0; r];
            row[j] = 1.0;
            return row;
        }
    }
    let mut row: Vec<f64> = (0..r).map(|j| wts[j] / (q - nodes[j])).collect();
    let sum: f64 = row.iter().sum();
    for x in row.iter_mut() {
        *x /= sum;
    }
    row
}

#[allow(clippy::too_many_arguments)]
fn separated_kernel(
    mu: ArrayView1<f64>,
    v: ArrayView2<f64>,
    d: ArrayView1<f64>,
    f_nodes: ArrayView2<f64>,
    w: ArrayView1<f64>,
    points: usize,
    rm: usize,
    rs_req: usize,
) -> (Array1<f64>, f64) {
    let n = mu.len();
    let q = f_nodes.nrows();
    let sd: Vec<f64> = d.iter().map(|x| x.sqrt()).collect();
    let sd_min = sd.iter().cloned().fold(f64::MAX, f64::min);
    let sd_max = sd.iter().cloned().fold(f64::MIN, f64::max);
    let rs = if sd_max - sd_min < 1e-12 { 1 } else { rs_req };

    let mut m_all = vec![0.0f64; q * n];
    let mut m_lo = f64::MAX;
    let mut m_hi = f64::MIN;
    for qi in 0..q {
        for i in 0..n {
            let mut mi = mu[i];
            for r in 0..v.ncols() {
                mi += v[[i, r]] * f_nodes[[qi, r]];
            }
            m_all[qi * n + i] = mi;
            m_lo = m_lo.min(mi);
            m_hi = m_hi.max(mi);
        }
    }
    let lo = m_lo - 8.0 * sd_max;
    let hi = m_hi + 8.0 * sd_max;
    let dx = (hi - lo) / (points - 1) as f64;

    let mn = cheb_nodes(m_lo, m_hi, rm);
    let sn = if rs == 1 {
        vec![0.5 * (sd_min + sd_max)]
    } else {
        cheb_nodes(sd_min, sd_max, rs)
    };
    let wm = bary_weights(&mn);
    let ws = bary_weights(&sn);
    let r_tot = rm * rs;

    // kernel tables at Chebyshev nodes: (r_tot, points)
    let mut logs_c = vec![0.0f64; r_tot * points];
    let mut haz_c = vec![0.0f64; r_tot * points];
    for cm in 0..rm {
        for cs in 0..rs {
            let c = cm * rs + cs;
            let inv_sd = 1.0 / sn[cs];
            let ln_c = sn[cs].ln() + LN_SQRT_2PI;
            for t in 0..points {
                let x = lo + t as f64 * dx;
                let z = (x - mn[cm]) * inv_sd;
                let ls = log_ndtr(-z);
                logs_c[c * points + t] = ls;
                haz_c[c * points + t] = (-0.5 * z * z - ln_c - ls).exp();
            }
        }
    }
    // per-runner sigma rows (fixed across nodes)
    let ts_rows: Vec<Vec<f64>> = (0..n).map(|i| bary_row(&sn, &ws, sd[i])).collect();

    let p: Vec<f64> = (0..q)
        .into_par_iter()
        .map(|qi| {
            let m = &m_all[qi * n..(qi + 1) * n];
            let wq = w[qi];
            // Tm rows and the aggregation matrix A[cm][cs] = sum_i Tm_i Ts_i
            let tm_rows: Vec<Vec<f64>> =
                (0..n).map(|i| bary_row(&mn, &wm, m[i])).collect();
            let mut amat = vec![0.0f64; r_tot];
            for i in 0..n {
                for (cm, tmv) in tm_rows[i].iter().enumerate() {
                    for (cs, tsv) in ts_rows[i].iter().enumerate() {
                        amat[cm * rs + cs] += tmv * tsv;
                    }
                }
            }
            // field(x) = sum_c amat_c logS_c(x); weights = exp(field) dx
            // b_c = sum_x haz_c(x) * weights(x)
            let mut b = vec![0.0f64; r_tot];
            for t in 0..points {
                let mut field = 0.0;
                for c in 0..r_tot {
                    field += amat[c] * logs_c[c * points + t];
                }
                if field > -745.0 {
                    let wt = field.exp() * dx;
                    for c in 0..r_tot {
                        b[c] += haz_c[c * points + t] * wt;
                    }
                }
            }
            // p_i = sum_c T_i(c) b_c
            let mut acc = vec![0.0f64; n];
            for i in 0..n {
                let mut s = 0.0;
                for (cm, tmv) in tm_rows[i].iter().enumerate() {
                    for (cs, tsv) in ts_rows[i].iter().enumerate() {
                        s += tmv * tsv * b[cm * rs + cs];
                    }
                }
                acc[i] = wq * s;
            }
            acc
        })
        .reduce(
            || vec![0.0f64; n],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        );

    let total: f64 = p.iter().sum();
    let out = Array1::from_iter(p.into_iter().map(|x| x / total));
    (out, total)
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


// ---- GHK baseline in Rust: like-for-like compiled comparison ------------

/// splitmix64 seeded xoshiro256++ (dependency-free PRNG).
struct Xo {
    s: [u64; 4],
}
impl Xo {
    fn new(seed: u64) -> Self {
        let mut x = seed;
        let mut s = [0u64; 4];
        for v in s.iter_mut() {
            x = x.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            *v = z ^ (z >> 31);
        }
        Xo { s }
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let r = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }
    #[inline]
    fn uniform(&mut self) -> f64 {
        // (0, 1): 53-bit mantissa, offset to avoid exact 0
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / 9007199254740992.0)
    }
}

/// Phi(z) via erfc.
#[inline]
fn ndtr(z: f64) -> f64 {
    0.5 * libm::erfc(-z * std::f64::consts::FRAC_1_SQRT_2)
}

/// Inverse normal CDF: Acklam's rational approximation plus one Halley
/// refinement through erfc, giving ~1e-15 accuracy.
fn ndtri(p: f64) -> f64 {
    let p = p.clamp(1e-300, 1.0 - 1e-16);
    const A: [f64; 6] = [-3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00];
    const B: [f64; 5] = [-5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01];
    const C: [f64; 6] = [-7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00, 2.938163982698783e+00];
    const D: [f64; 4] = [7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00];
    let x = if p < 0.02425 {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 0.97575 {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    // one Halley step: f = Phi(x) - p
    let e = ndtr(x) - p;
    let u = e * (2.0 * std::f64::consts::PI).sqrt() * (0.5 * x * x).exp();
    x - u / (1.0 + 0.5 * x * u)
}

fn cholesky(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if d <= 0.0 {
            return false;
        }
        let dj = d.sqrt();
        a[j * n + j] = dj;
        for i in (j + 1)..n {
            let mut v = a[i * n + j];
            for k in 0..j {
                v -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = v / dj;
        }
        for k in (j + 1)..n {
            a[j * n + k] = 0.0;
        }
    }
    true
}

/// GHK estimate of P(alternative i has the max utility), R draws.
fn ghk_prob_one(mu: &[f64], sigma: &[f64], n: usize, i: usize, r_draws: usize,
                seed: u64) -> f64 {
    let m = n - 1;
    // difference covariance C = M Sigma M', means a = mu_others - mu_i
    let others: Vec<usize> = (0..n).filter(|&j| j != i).collect();
    let mut a = vec![0.0f64; m];
    let mut c = vec![0.0f64; m * m];
    for (r_, &jr) in others.iter().enumerate() {
        a[r_] = mu[jr] - mu[i];
        for (c_, &jc) in others.iter().enumerate() {
            c[r_ * m + c_] = sigma[jr * n + jc] - sigma[jr * n + i]
                - sigma[i * n + jc] + sigma[i * n + i];
        }
    }
    for d in 0..m {
        c[d * m + d] += 1e-12;
    }
    if !cholesky(&mut c, m) {
        return f64::NAN;
    }
    let mut rng = Xo::new(seed);
    // draw-major layout: z[t] is an R-vector; the running mean for step t is
    // accumulated with R-length axpy updates (SIMD-friendly), matching the
    // vectorization structure of the NumPy baseline
    let mut z = vec![vec![0.0f64; r_draws]; m];
    let mut mean = vec![0.0f64; r_draws];
    let mut logprob = vec![0.0f64; r_draws];
    for t in 0..m {
        mean[..r_draws].fill(0.0);
        for k in 0..t {
            let ltk = c[t * m + k];
            if ltk != 0.0 {
                let zk = &z[k];
                for (mv, zv) in mean.iter_mut().zip(zk.iter()) {
                    *mv += ltk * zv;
                }
            }
        }
        let ltt = c[t * m + t];
        let at = a[t];
        let zt = &mut z[t];
        for dr in 0..r_draws {
            let b = (-at - mean[dr]) / ltt;
            let fb = ndtr(b);
            logprob[dr] += fb.max(1e-300).ln();
            let u = rng.uniform() * fb;
            zt[dr] = ndtri(u.clamp(1e-300, 1.0 - 1e-16));
        }
    }
    let mut acc = 0.0f64;
    for lp in logprob.iter() {
        acc += lp.exp();
    }
        acc / r_draws as f64
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

#[pymodule]
fn fastrace(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(forward_and_slopes, m)?)?;
    m.add_function(wrap_pyfunction!(jacobian_vector_product, m)?)?;
    m.add_function(wrap_pyfunction!(win_probabilities_factor_separated, m)?)?;
    m.add_function(wrap_pyfunction!(ghk_all_shares, m)?)?;
    m.add_function(wrap_pyfunction!(win_probabilities_factor, m)?)?;
    Ok(())
}
