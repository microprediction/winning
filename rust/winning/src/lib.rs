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

pub use ndarray;
use ndarray::{Array1, ArrayView1, ArrayView2};
use rayon::prelude::*;

pub const TILE: usize = 256;
pub const LN_SQRT_2PI: f64 = 0.918938533204672741780329736406;

/// log(Phi(z)). libm::erfc is exact to double precision until it
/// underflows (z < about -37.5); beyond that use the asymptotic series
/// log Phi(z) = -z^2/2 - log(-z) - log sqrt(2 pi) + log(1 - 1/z^2 + 3/z^4 - 15/z^6).
pub fn log_ndtr(z: f64) -> f64 {
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
pub fn forward_kernel(
    mu: ArrayView1<f64>,
    v: ArrayView2<f64>,
    d: ArrayView1<f64>,
    f_nodes: ArrayView2<f64>,
    w: ArrayView1<f64>,
    points: usize,
    lo_in: f64,
    hi_in: f64,
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
    if lo_in.is_finite() && hi_in.is_finite() && hi_in > lo_in {
        lo = lo_in;
        hi = hi_in;
    }
    let dx = (hi - lo) / (points - 1) as f64;

    // per-task scratch is n * tile floats twice over; cap it so large n
    // does not swap (n = 1e6 with the fixed 256-tile costs 2 GB per task)
    let tile = TILE.min((25_000_000usize / n.max(1)).max(4));
    let p: Vec<f64> = (0..q)
        .into_par_iter()
        .map(|qi| {
            let m = &m_all[qi * n..(qi + 1) * n];
            let wq = w[qi];
            let mut acc = vec![0.0f64; 2 * n];
            let mut logs = vec![0.0f64; n * tile];
            let mut logg = vec![0.0f64; n * tile];
            let mut field = vec![0.0f64; tile];
            let mut t0 = 0;
            while t0 < points {
                let tl = tile.min(points - t0);
                field[..tl].fill(0.0);
                for i in 0..n {
                    let inv_sd = 1.0 / sd[i];
                    let ln_i = log_norm[i];
                    let mi = m[i];
                    let row_s = &mut logs[i * tile..i * tile + tl];
                    let row_g = &mut logg[i * tile..i * tile + tl];
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
                    let row_s = &logs[i * tile..i * tile + tl];
                    let row_g = &logg[i * tile..i * tile + tl];
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


#[allow(clippy::too_many_arguments)]
pub fn jvp_kernel(
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


pub fn cheb_nodes(a: f64, b: f64, r: usize) -> Vec<f64> {
    (0..r)
        .map(|k| 0.5 * (a + b)
            + 0.5 * (b - a)
                * ((2 * k + 1) as f64 * std::f64::consts::PI / (2 * r) as f64).cos())
        .collect()
}

pub fn bary_weights(nodes: &[f64]) -> Vec<f64> {
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
pub fn bary_row(nodes: &[f64], wts: &[f64], q: f64) -> Vec<f64> {
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
pub fn separated_kernel(
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


// ---- GHK baseline in Rust: like-for-like compiled comparison ------------

/// splitmix64 seeded xoshiro256++ (dependency-free PRNG).
pub struct Xo {
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
pub fn ndtr(z: f64) -> f64 {
    0.5 * libm::erfc(-z * std::f64::consts::FRAC_1_SQRT_2)
}

/// Inverse normal CDF: Acklam's rational approximation plus one Halley
/// refinement through erfc, giving ~1e-15 accuracy.
pub fn ndtri(p: f64) -> f64 {
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

pub fn cholesky(a: &mut [f64], n: usize) -> bool {
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
pub fn ghk_prob_one(mu: &[f64], sigma: &[f64], n: usize, i: usize, r_draws: usize,
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


// ---------------------------------------------------------------------------
// block race: clustered covariance (Schur rung 1 of research/pqrace)
// ---------------------------------------------------------------------------

/// Win probabilities under the nested-effects model (MAX-wins convention):
///   Y_i = mu[i] + v[i] * a_{cluster[i]} + sd[i] * eps_i,
/// a_c ~ N(0,1) iid across clusters, each integrated by the supplied 1-d
/// quadrature. `starts` gives the first member index of each cluster;
/// members must be sorted by cluster (the Python wrapper sorts/unpermutes).
///
/// The Schur move: across-cluster independence factorizes the field by
/// cluster, G(x) = prod_c G_c(x); the winner's own cluster is handled by
/// leave-one-out inside its block. Parallel over lattice columns, fully
/// fused per column -- no N x L temporaries.

/// Scratch threshold for the fast path: per-thread lf/pf scratch is
/// 2 * n * qa * 8 bytes; beyond ~10M entries the streaming kernel's O(max
/// cluster) memory wins over its ~35% arithmetic premium.
pub const FAST_SCRATCH_ENTRIES: usize = 10_000_000;

pub fn block_kernel(
    mu: ArrayView1<f64>,
    sd: ArrayView1<f64>,
    v: ArrayView1<f64>,
    starts: &[usize],
    a_nodes: ArrayView1<f64>,
    a_w: ArrayView1<f64>,
    points: usize,
    lo_in: f64,
    hi_in: f64,
    fast_max_entries: usize,
) -> Array1<f64> {
    let n = mu.len();
    let qa = a_nodes.len();
    if n * qa > fast_max_entries {
        return block_kernel_streaming(mu, sd, v, starts, a_nodes, a_w,
                                      points, lo_in, hi_in);
    }
    let n_c = starts.len();
    let (lo, hi) = if lo_in.is_finite() && hi_in.is_finite() {
        (lo_in, hi_in)
    } else {
        let mut lo = f64::MAX;
        let mut hi = f64::MIN;
        let amax = a_nodes.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        for i in 0..n {
            let spread = 8.0 * sd[i] + amax * v[i].abs();
            lo = lo.min(mu[i] - spread);
            hi = hi.max(mu[i] + spread);
        }
        (lo, hi)
    };
    let dx = (hi - lo) / (points - 1) as f64;
    let p: Vec<f64> = (0..points)
        .into_par_iter()
        .fold(
            || vec![0.0f64; n],
            |mut acc, t| {
                let x = lo + dx * t as f64;
                let mut lf = vec![0.0f64; n * qa];
                let mut pf = vec![0.0f64; n * qa];
                let mut s_ca = vec![0.0f64; n_c * qa];
                for c in 0..n_c {
                    let e = if c + 1 < n_c { starts[c + 1] } else { n };
                    for a in 0..qa {
                        let mut s_sum = 0.0;
                        for i in starts[c]..e {
                            let z = (x - mu[i] - v[i] * a_nodes[a]) / sd[i];
                            let l = log_ndtr(z);
                            lf[i * qa + a] = l;
                            pf[i * qa + a] =
                                (-0.5 * z * z - LN_SQRT_2PI - sd[i].ln()).exp();
                            s_sum += l;
                        }
                        s_ca[c * qa + a] = s_sum;
                    }
                }
                let mut log_g = vec![0.0f64; n_c];
                let mut log_all = 0.0;
                for c in 0..n_c {
                    let mut g = 0.0;
                    for a in 0..qa {
                        g += a_w[a] * s_ca[c * qa + a].exp();
                    }
                    let lg = if g > 1e-300 { g.ln() } else { -690.0 };
                    log_g[c] = lg;
                    log_all += lg;
                }
                for c in 0..n_c {
                    let e = if c + 1 < n_c { starts[c + 1] } else { n };
                    let rest = log_all - log_g[c];
                    if rest < -690.0 {
                        continue;
                    }
                    let rest_e = rest.exp();
                    for i in starts[c]..e {
                        let mut h = 0.0;
                        for a in 0..qa {
                            let ex = s_ca[c * qa + a] - lf[i * qa + a];
                            if ex > -690.0 {
                                h += a_w[a] * pf[i * qa + a] * ex.exp();
                            }
                        }
                        acc[i] += h * rest_e;
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0.0f64; n],
            |mut a, b| {
                for i in 0..n {
                    a[i] += b[i];
                }
                a
            },
        );
    Array1::from_iter(p.into_iter().map(|x| (x * dx).max(0.0)))
}

pub fn block_kernel_streaming(
    mu: ArrayView1<f64>,
    sd: ArrayView1<f64>,
    v: ArrayView1<f64>,
    starts: &[usize],
    a_nodes: ArrayView1<f64>,
    a_w: ArrayView1<f64>,
    points: usize,
    lo_in: f64,
    hi_in: f64,
) -> Array1<f64> {
    // STREAMING kernel: per (column, cluster) only that cluster's logF/pdf
    // scratch is held (max_cluster x qa), so memory is O(largest cluster),
    // not O(N) -- the model is input-bound, not scratch-bound. Costs one
    // extra pass of arithmetic per column; measured competitive because the
    // scratch now lives in cache.
    let n = mu.len();
    let qa = a_nodes.len();
    let n_c = starts.len();
    let mut maxc = 0usize;
    for c in 0..n_c {
        let e = if c + 1 < n_c { starts[c + 1] } else { n };
        maxc = maxc.max(e - starts[c]);
    }
    let (lo, hi) = if lo_in.is_finite() && hi_in.is_finite() {
        (lo_in, hi_in)
    } else {
        let mut lo = f64::MAX;
        let mut hi = f64::MIN;
        let amax = a_nodes.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
        for i in 0..n {
            let spread = 8.0 * sd[i] + amax * v[i].abs();
            lo = lo.min(mu[i] - spread);
            hi = hi.max(mu[i] + spread);
        }
        (lo, hi)
    };
    let dx = (hi - lo) / (points - 1) as f64;

    let p: Vec<f64> = (0..points)
        .into_par_iter()
        .fold(
            || vec![0.0f64; n],
            |mut acc, t| {
                let x = lo + dx * t as f64;
                let mut lf = vec![0.0f64; maxc * qa];
                let mut pf = vec![0.0f64; maxc * qa];
                let mut s_a = vec![0.0f64; qa];
                let mut log_g = vec![0.0f64; n_c];
                let mut log_all = 0.0;
                // pass 1: per-cluster fields (cluster scratch reused)
                for c in 0..n_c {
                    let e = if c + 1 < n_c { starts[c + 1] } else { n };
                    for a in 0..qa {
                        s_a[a] = 0.0;
                    }
                    for i in starts[c]..e {
                        for a in 0..qa {
                            let z = (x - mu[i] - v[i] * a_nodes[a]) / sd[i];
                            s_a[a] += log_ndtr(z);
                        }
                    }
                    let mut g = 0.0;
                    for a in 0..qa {
                        g += a_w[a] * s_a[a].exp();
                    }
                    let lg = if g > 1e-300 { g.ln() } else { -690.0 };
                    log_g[c] = lg;
                    log_all += lg;
                }
                // pass 2: member terms, one cluster's scratch at a time
                for c in 0..n_c {
                    let e = if c + 1 < n_c { starts[c + 1] } else { n };
                    let rest = log_all - log_g[c];
                    if rest < -690.0 {
                        continue;
                    }
                    let rest_e = rest.exp();
                    for a in 0..qa {
                        s_a[a] = 0.0;
                    }
                    for (k, i) in (starts[c]..e).enumerate() {
                        for a in 0..qa {
                            let z = (x - mu[i] - v[i] * a_nodes[a]) / sd[i];
                            let l = log_ndtr(z);
                            lf[k * qa + a] = l;
                            pf[k * qa + a] =
                                (-0.5 * z * z - LN_SQRT_2PI - sd[i].ln()).exp();
                            s_a[a] += l;
                        }
                    }
                    for (k, i) in (starts[c]..e).enumerate() {
                        let mut h = 0.0;
                        for a in 0..qa {
                            let ex = s_a[a] - lf[k * qa + a];
                            if ex > -690.0 {
                                h += a_w[a] * pf[k * qa + a] * ex.exp();
                            }
                        }
                        acc[i] += h * rest_e;
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0.0f64; n],
            |mut a, b| {
                for i in 0..n {
                    a[i] += b[i];
                }
                a
            },
        );
    Array1::from_iter(p.into_iter().map(|x| (x * dx).max(0.0)))
}


/// Rank-r block race: like `block_race`, but each cluster's shared effect is
/// r-dimensional with a free per-member loading MATRIX v (n x r); the
/// quadrature nodes are (Q x r) with weights w. Members must be sorted by
/// cluster; `starts` gives each cluster's first index. MAX-wins.
pub fn block_kernel_r(
    mu: ArrayView1<f64>,
    sd: ArrayView1<f64>,
    v: ArrayView2<f64>,
    starts: &[usize],
    nodes: ArrayView2<f64>,
    w: ArrayView1<f64>,
    points: usize,
    lo_in: f64,
    hi_in: f64,
) -> Array1<f64> {
    let n = mu.len();
    let qa = nodes.nrows();
    let r = nodes.ncols();
    let n_c = starts.len();
    // shift[i*qa + a] = sum_k v[i,k] nodes[a,k], computed once (r is small)
    let mut shift = vec![0.0f64; n * qa];
    for i in 0..n {
        for a in 0..qa {
            let mut s_ = 0.0;
            for k in 0..r {
                s_ += v[[i, k]] * nodes[[a, k]];
            }
            shift[i * qa + a] = s_;
        }
    }
    let (lo, hi) = if lo_in.is_finite() && hi_in.is_finite() {
        (lo_in, hi_in)
    } else {
        let mut lo = f64::MAX;
        let mut hi = f64::MIN;
        for i in 0..n {
            let mut smax: f64 = 0.0;
            for a in 0..qa {
                smax = smax.max(shift[i * qa + a].abs());
            }
            lo = lo.min(mu[i] - 8.0 * sd[i] - smax);
            hi = hi.max(mu[i] + 8.0 * sd[i] + smax);
        }
        (lo, hi)
    };
    let dx = (hi - lo) / (points - 1) as f64;

    let p: Vec<f64> = (0..points)
        .into_par_iter()
        .fold(
            || vec![0.0f64; n],
            |mut acc, t| {
                let x = lo + dx * t as f64;
                let mut lf = vec![0.0f64; n * qa];
                let mut pf = vec![0.0f64; n * qa];
                let mut s_ca = vec![0.0f64; n_c * qa];
                for c in 0..n_c {
                    let e = if c + 1 < n_c { starts[c + 1] } else { n };
                    for a in 0..qa {
                        let mut s_sum = 0.0;
                        for i in starts[c]..e {
                            let z = (x - mu[i] - shift[i * qa + a]) / sd[i];
                            let l = log_ndtr(z);
                            lf[i * qa + a] = l;
                            pf[i * qa + a] =
                                (-0.5 * z * z - LN_SQRT_2PI - sd[i].ln()).exp();
                            s_sum += l;
                        }
                        s_ca[c * qa + a] = s_sum;
                    }
                }
                let mut log_g = vec![0.0f64; n_c];
                let mut log_all = 0.0;
                for c in 0..n_c {
                    let mut g = 0.0;
                    for a in 0..qa {
                        g += w[a] * s_ca[c * qa + a].exp();
                    }
                    let lg = if g > 1e-300 { g.ln() } else { -690.0 };
                    log_g[c] = lg;
                    log_all += lg;
                }
                for c in 0..n_c {
                    let e = if c + 1 < n_c { starts[c + 1] } else { n };
                    let rest = log_all - log_g[c];
                    if rest < -690.0 {
                        continue;
                    }
                    let rest_e = rest.exp();
                    for i in starts[c]..e {
                        let mut h = 0.0;
                        for a in 0..qa {
                            let ex = s_ca[c * qa + a] - lf[i * qa + a];
                            if ex > -690.0 {
                                h += w[a] * pf[i * qa + a] * ex.exp();
                            }
                        }
                        acc[i] += h * rest_e;
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0.0f64; n],
            |mut a, b| {
                for i in 0..n {
                    a[i] += b[i];
                }
                a
            },
        );
    Array1::from_iter(p.into_iter().map(|x| (x * dx).max(0.0)))
}

// ---------------------------------------------------------------------------
// Tree race: hierarchy of uniform shared effects, two message passes on the
// lattice. Port of winning.factor.blocks.tree_race_probabilities (the python
// wrapper negates, sorts by cluster, computes the bulk window, and
// normalizes; this kernel takes the sorted arrays and returns raw p).
// ---------------------------------------------------------------------------

pub const LN_TINY: f64 = -690.77552789821368; // ln(1e-300)

/// Evaluate g at x + delta on the uniform grid [lo, lo+dx, ...] (linear,
/// clamped at the edges) -- matches np.interp(x, x - delta, g, g[0], g[-1]).
pub fn interp_shift(g: &[f64], delta_over_dx: f64) -> Vec<f64> {
    let m = g.len();
    (0..m)
        .map(|t| {
            let pos = t as f64 + delta_over_dx;
            if pos <= 0.0 {
                g[0]
            } else if pos >= (m - 1) as f64 {
                g[m - 1]
            } else {
                let k = pos.floor() as usize;
                let r = pos - k as f64;
                g[k] * (1.0 - r) + g[k + 1] * r
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn tree_kernel(
    mu: &[f64],
    sd: &[f64],
    v: &[f64],
    starts: &[usize],
    parent: &[i64],
    lam: &[f64],
    an: &[f64],
    aw: &[f64],
    points: usize,
    lo: f64,
    hi: f64,
) -> Vec<f64> {
    let n = mu.len();
    let nc = starts.len();
    let nt = parent.len();
    let qa = an.len();
    let dx = (hi - lo) / (points - 1) as f64;
    let mut ends: Vec<usize> = starts[1..].to_vec();
    ends.push(n);

    // pass 1: per-cluster survival logs S[c][q][t]
    let mut s_arr = vec![0.0f64; nc * qa * points];
    s_arr
        .par_chunks_mut(qa * points)
        .enumerate()
        .for_each(|(c, sc)| {
            for i in starts[c]..ends[c] {
                let inv_sd = 1.0 / sd[i];
                for q in 0..qa {
                    let mi = mu[i] + v[i] * an[q];
                    let row = &mut sc[q * points..(q + 1) * points];
                    for (t, rt) in row.iter_mut().enumerate() {
                        let x = lo + t as f64 * dx;
                        let z = (x - mi) * inv_sd;
                        *rt += log_ndtr(z).max(LN_TINY);
                    }
                }
            }
        });

    // leaf messages G[c] = sum_q aw_q exp(min(S, 0))
    let mut g: Vec<Vec<f64>> = vec![vec![0.0; points]; nt];
    for c in 0..nc {
        let gc = &mut g[c];
        for q in 0..qa {
            let row = &s_arr[c * qa * points + q * points..c * qa * points + (q + 1) * points];
            for t in 0..points {
                gc[t] += aw[q] * row[t].min(0.0).exp();
            }
        }
    }

    // tree bookkeeping (matches the python): traversal order must be TREE
    // depth in hops, not the |lam| path sum — zero strengths (from_linkage's
    // floored merges) tie the path sums and a tied sort visits children
    // before their parents, reading cavities still at their initial value.
    let mut depth_hops = vec![0usize; nt];
    for t in 0..nt {
        let mut d_ = 0usize;
        let mut u = t;
        while parent[u] >= 0 {
            d_ += 1;
            u = parent[u] as usize;
        }
        depth_hops[t] = d_;
    }
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); nt];
    for t in 0..nt {
        if parent[t] >= 0 {
            children[parent[t] as usize].push(t);
        }
    }

    // upward pass: internal nodes, deepest first (stable on ties)
    let mut up: Vec<usize> = (nc..nt).collect();
    up.sort_by_key(|&t| std::cmp::Reverse(depth_hops[t]));
    for &t in &up {
        let mut acc = vec![0.0f64; points];
        for q in 0..qa {
            let mut prod = vec![1.0f64; points];
            for &c in &children[t] {
                let sh = interp_shift(&g[c], lam[t] * an[q] / dx);
                for (p_, s_) in prod.iter_mut().zip(sh) {
                    *p_ *= s_;
                }
            }
            for (a_, p_) in acc.iter_mut().zip(prod) {
                *a_ += aw[q] * p_;
            }
        }
        for a_ in acc.iter_mut() {
            *a_ = a_.max(0.0);
        }
        g[t] = acc;
    }

    // downward pass: shallowest first (stable on ties)
    let mut r: Vec<Vec<f64>> = vec![vec![1.0; points]; nt];
    let mut down: Vec<usize> = (0..nt).collect();
    down.sort_by_key(|&t| depth_hops[t]);
    for &t in &down {
        if parent[t] < 0 {
            continue;
        }
        let pa = parent[t] as usize;
        let mut sm = vec![0.0f64; points];
        for q in 0..qa {
            let sh = interp_shift(&r[pa], -lam[pa] * an[q] / dx);
            for (s_, v_) in sm.iter_mut().zip(sh) {
                *s_ += aw[q] * v_;
            }
        }
        let mut prod = vec![1.0f64; points];
        for &s_ in &children[pa] {
            if s_ != t {
                for (p_, gv) in prod.iter_mut().zip(&g[s_]) {
                    *p_ *= gv;
                }
            }
        }
        let rt: Vec<f64> = sm
            .iter()
            .zip(prod)
            .map(|(s_, p_)| (s_ * p_).max(0.0))
            .collect();
        r[t] = rt;
    }

    // pass 2: per-runner win integrand against its own leaf message
    let cluster_of: Vec<usize> = {
        let mut cl = vec![0usize; n];
        for c in 0..nc {
            for i in starts[c]..ends[c] {
                cl[i] = c;
            }
        }
        cl
    };
    (0..n)
        .into_par_iter()
        .map(|i| {
            let c = cluster_of[i];
            let inv_sd = 1.0 / sd[i];
            let ln_i = sd[i].ln() + LN_SQRT_2PI;
            let rc = &r[c];
            let mut pi = 0.0f64;
            for q in 0..qa {
                let mi = mu[i] + v[i] * an[q];
                let row =
                    &s_arr[c * qa * points + q * points..c * qa * points + (q + 1) * points];
                for t in 0..points {
                    let x = lo + t as f64 * dx;
                    let z = (x - mi) * inv_sd;
                    let lf = log_ndtr(z).max(LN_TINY);
                    let e = (row[t] - lf).min(0.0) + (-0.5 * z * z - ln_i);
                    if e > -745.0 {
                        pi += aw[q] * e.exp() * rc[t];
                    }
                }
            }
            (pi * dx).max(0.0)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Classic core: the original state-price lattice calibration
// (winning.lattice / winning.lattice_calibration), dead-heat multiplicity
// machinery included, with the reference implementation's exact epsilon
// conventions so the two paths agree to numerical noise.
// ---------------------------------------------------------------------------

pub fn pdf_to_cdf(f: &[f64]) -> Vec<f64> {
    let mut c = Vec::with_capacity(f.len());
    let mut s = 0.0;
    for &x in f {
        s += x;
        c.push(s);
    }
    c
}

pub fn cdf_to_pdf(c: &[f64]) -> Vec<f64> {
    let mut f = Vec::with_capacity(c.len());
    let mut prev = 0.0;
    for &x in c {
        f.push(x - prev);
        prev = x;
    }
    f
}

pub fn integer_shift(cdf: &[f64], k: i64) -> Vec<f64> {
    let m = cdf.len() as i64;
    let k = k.clamp(-(m - 1), m - 1);
    if k < 0 {
        let a = (-k) as usize;
        let last = cdf[cdf.len() - 1];
        let mut out: Vec<f64> = cdf[a..].to_vec();
        out.extend(std::iter::repeat(last).take(a));
        out
    } else if k == 0 {
        cdf.to_vec()
    } else {
        let a = k as usize;
        let mut out = vec![0.0; a];
        out.extend_from_slice(&cdf[..cdf.len() - a]);
        out
    }
}

pub fn low_high(offset: f64, l: i64) -> ((i64, f64), (i64, f64)) {
    let lf = l as f64;
    if offset > -lf + 2.0 && offset < lf - 2.0 {
        let lo = offset.floor() as i64;
        let up = offset.ceil() as i64;
        let r = offset - lo as f64;
        ((lo, 1.0 - r), (up, r))
    } else if offset >= lf - 2.0 {
        ((l - 2, 1.0), (l - 1, 0.0))
    } else {
        ((-l + 1, 0.0), (-l + 2, 1.0))
    }
}

pub fn shifted_cdf(cdf: &[f64], offset: f64, l: i64) -> Vec<f64> {
    let ((a, ac), (b, bc)) = low_high(offset, l);
    let sa = integer_shift(cdf, a);
    let sb = integer_shift(cdf, b);
    sa.iter().zip(sb).map(|(x, y)| ac * x + bc * y).collect()
}

/// Fold the field: density/cdf/multiplicity of the minimum, dead heats
/// tracked via the reference _winner_of_two_pdf recursion.
pub fn winner_of_many(cdfs: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let m = cdfs[0].len();
    let mut cdf_min = cdfs[0].clone();
    let mut mult = vec![1.0f64; m];
    for cb in &cdfs[1..] {
        let fa = cdf_to_pdf(&cdf_min);
        let fb = cdf_to_pdf(cb);
        let mut new_cdf = Vec::with_capacity(m);
        let mut new_mult = Vec::with_capacity(m);
        for t in 0..m {
            let win = fa[t] * (1.0 - cb[t]);
            let draw = fa[t] * fb[t];
            let lose = fb[t] * (1.0 - cdf_min[t]);
            new_mult.push(
                (win * mult[t] + draw * (mult[t] + 1.0) + lose * 1.0 + 1e-18)
                    / (win + draw + lose + 1e-18),
            );
            new_cdf.push(1.0 - (1.0 - cdf_min[t]) * (1.0 - cb[t]));
        }
        cdf_min = new_cdf;
        mult = new_mult;
    }
    (cdf_min, mult)
}

/// get_the_rest + conditional payoff, summed: the expected payoff of a
/// contestant with cdf `cdf` against the field (cdf_all, mult_all).
pub fn expected_payoff_sum(cdf: &[f64], cdf_all: &[f64], mult_all: &[f64]) -> f64 {
    let m = cdf.len();
    let f1 = cdf_to_pdf(cdf);
    let mut cdf_rest = Vec::with_capacity(m);
    for t in 0..m {
        let s = 1.0 - cdf_all[t];
        let s1 = 1.0 - cdf[t];
        cdf_rest.push(1.0 - (s + 1e-18) / (s1 + 1e-6));
    }
    let f_rest = cdf_to_pdf(&cdf_rest);
    // multiplicity of the rest: left-tail inversion, right-tail asymptotic,
    // switch at the mode of f1 (first argmax), exactly as the reference
    let mut kmax = 0;
    let mut fmax = f64::MIN;
    for (t, &x) in f1.iter().enumerate() {
        if x > fmax {
            fmax = x;
            kmax = t;
        }
    }
    let mut mult_rest = Vec::with_capacity(m);
    for t in 0..m {
        let mm = mult_all[t];
        let s1 = 1.0 - cdf[t];
        let srest = (1.0 - cdf_all[t] + 1e-18) / (s1 + 1e-6);
        if t < kmax {
            let numer =
                mm * f1[t] * srest + mm * (f1[t] + s1) * f_rest[t] - f1[t] * (srest + f_rest[t]);
            let denom = f_rest[t] * (f1[t] + s1);
            mult_rest.push((1e-18 + numer) / (1e-18 + denom));
        } else {
            let t1 = (s1 + 1e-18) / (f1[t] + 1e-6);
            let trest = (srest + 1e-18) / (f_rest[t] + 1e-6);
            mult_rest.push(mm * trest / (1.0 + t1) + mm - (1.0 + trest) / (1.0 + t1));
        }
    }
    // forced monotone cdf of the rest, then payoff = win + draw/(1 + mult)
    let mut run = f64::MIN;
    let mut total = 0.0;
    let mut prev = 0.0;
    for t in 0..m {
        run = run.max(cdf_rest[t]);
        let fr = run - prev;
        prev = run;
        total += f1[t] * (1.0 - run) + f1[t] * fr / (1.0 + mult_rest[t]);
    }
    total
}

/// implicit_state_prices: expected payoff of the base density shifted to
/// each offset (float offsets blend the two integer shifts).
pub fn implicit_prices(
    base_cdf: &[f64],
    cdf_all: &[f64],
    mult_all: &[f64],
    offsets: &[f64],
    l: i64,
) -> Vec<f64> {
    offsets
        .par_iter()
        .map(|&k| {
            if k == k.trunc() {
                expected_payoff_sum(&integer_shift(base_cdf, k as i64), cdf_all, mult_all)
            } else {
                let ((a, ac), (b, bc)) = low_high(k, l);
                ac * expected_payoff_sum(&integer_shift(base_cdf, a), cdf_all, mult_all)
                    + bc * expected_payoff_sum(&integer_shift(base_cdf, b), cdf_all, mult_all)
            }
        })
        .collect()
}

/// np.interp(x, xp, fp) for ascending xp, end-clamped.
pub fn interp1(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    if x <= xp[0] {
        return fp[0];
    }
    let last = xp.len() - 1;
    if x >= xp[last] {
        return fp[last];
    }
    let j = xp.partition_point(|&p| p <= x) - 1;
    if j >= last {
        return fp[last];
    }
    let denom = xp[j + 1] - xp[j];
    if denom <= 0.0 {
        return fp[j];
    }
    fp[j] + (x - xp[j]) / denom * (fp[j + 1] - fp[j])
}
