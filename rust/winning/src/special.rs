//! Tail-accurate special functions for the custom-base kernels.
//!
//! The NumPy reference evaluates these bases with SciPy; enabling the
//! compiled path must not change a legal model. Three places where the
//! general-purpose crates (puruspe, the bare owens-t identity) were not
//! good enough are handled here (#134, #193):
//!
//! - `gamma_q`: regularized upper incomplete gamma Q(a, x). puruspe's
//!   gammq returned Q = 1 for tiny x and Q = 0 well before underflow for
//!   small a (Q(0.02, 30) = 6.5e-17 came back 0; Q(0.02, 6e-28) = 0.71
//!   came back 1), which is exactly the exponential-power base at large
//!   beta: a = 1/beta is small and t = |z/a|^beta spans both extremes.
//! - `student_sf`: puruspe's betai forms 1 - x internally, which at
//!   x = nu/(nu + t^2) loses every digit once nu is large (0.4% off at
//!   nu = 1e12, uniform races at 1e50, a panic at inf).
//! - `skew_normal_sf`: S = Phi(-x) + 2 T(x, alpha) cancels two O(1e-1)
//!   terms to an O(1e-26) answer for negative alpha.

use crate::{log_ndtr, ndtr};

/// Regularized upper incomplete gamma Q(a, x), a > 0, x >= 0.
/// Series for P when x < a + 1, Lentz continued fraction for Q
/// otherwise (Numerical Recipes 6.2), with the prefactor
/// exp(-x + a ln x - lnGamma(a)) formed in the log domain.
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x.is_infinite() {
        return 0.0;
    }
    let lpre = -x + a * x.ln() - libm::lgamma(a);
    if x < a + 1.0 {
        // P = pre * sum_n x^n / (a (a+1) ... (a+n))
        let mut ap = a;
        let mut del = 1.0 / a;
        let mut sum = del;
        for _ in 0..10_000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-17 {
                break;
            }
        }
        let p = (lpre + sum.ln()).exp();
        (1.0 - p).max(0.0)
    } else {
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..10_000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-16 {
                break;
            }
        }
        (lpre + h.ln()).exp()
    }
}

/// log Gamma(a + 1/2) - log Gamma(a). The plain lgamma difference
/// cancels at large a (lgamma(5e49) ~ 5.6e51), so from a = 50 use the
/// asymptotic series (1/2) log a - 1/(8a) + 1/(192 a^3) - 1/(640 a^5),
/// whose truncation is below 1e-16 there.
pub fn lgamma_half_ratio(a: f64) -> f64 {
    if a >= 50.0 {
        let a3 = a * a * a;
        return 0.5 * a.ln() - 1.0 / (8.0 * a) + 1.0 / (192.0 * a3)
            - 1.0 / (640.0 * a3 * a * a);
    }
    libm::lgamma(a + 0.5) - libm::lgamma(a)
}

/// Continued fraction for the incomplete beta (Numerical Recipes 6.4,
/// modified Lentz).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..100_000 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

/// Student-t survival P(T > x) for nu > 0 degrees of freedom, via
/// I_w(nu/2, 1/2) with w = nu/(nu + x^2) AND its complement
/// 1 - w = x^2/(nu + x^2) both formed directly: puruspe's betai took w
/// alone and subtracted, which at x = 1e-3 was already 0.13% off at
/// nu = 30 and 55% off at nu = 1000. nu >= 1e4: Hill's normal transform
/// (CACM algorithm 395), measured against SciPy to ~2e-13 relative out
/// to sf ~ 1e-197 and convergent to the normal limit.
pub fn student_log_sf(x: f64, nu: f64) -> f64 {
    if nu >= 1e4 {
        let a = nu - 0.5;
        let b = 48.0 * a * a;
        let y = a * (x * x / nu).ln_1p();
        let y = (((((-0.4 * y - 3.3) * y - 24.0) * y - 85.5)
            / (0.8 * y * y + 100.0 + b) + y + 3.0) / b + 1.0) * y.sqrt();
        let z = if x >= 0.0 { y } else { -y };
        return log_ndtr(-z);
    }
    if x == 0.0 {
        return -std::f64::consts::LN_2;
    }
    let (a, b) = (0.5 * nu, 0.5);
    let x2 = x * x;
    let lw = -(x2 / nu).ln_1p();                 // log(nu/(nu+x^2))
    let lv = x2.ln() - (nu + x2).ln();           // log(x^2/(nu+x^2))
    let w = lw.exp();
    let v = lv.exp();
    // log of x^a (1-x)^b / (a B(a, b)) without the cancelling lgammas
    let lbt = lgamma_half_ratio(a) - 0.5 * std::f64::consts::PI.ln()
        + a * lw + b * lv;
    // tail = P(|T| > |x|) = I_w(a, b); each branch keeps its small side
    // exact and forms the other by one subtraction from 1
    let (tail, small_is_tail) = if w < (a + 1.0) / (a + b + 2.0) {
        ((lbt + betacf(a, b, w).ln() - a.ln()).exp(), true)
    } else {
        ((lbt + betacf(b, a, v).ln() - b.ln()).exp(), false)
    };
    let half_tail = if small_is_tail { 0.5 * tail } else { 0.5 * (1.0 - tail) };
    if x > 0.0 {
        half_tail.max(1e-300).ln()
    } else {
        (-half_tail).ln_1p()
    }
}

/// Student-t survival, exp(student_log_sf).
pub fn student_sf(x: f64, nu: f64) -> f64 {
    student_log_sf(x, nu).exp()
}

/// log density normalizer of the Student-t, log Gamma((nu+1)/2) -
/// log Gamma(nu/2) - log(nu pi)/2, cancellation-free at large nu.
pub fn student_log_norm(nu: f64) -> f64 {
    lgamma_half_ratio(0.5 * nu) - 0.5 * (nu * std::f64::consts::PI).ln()
}

/// 16-point Gauss-Legendre on [-1, 1] (nodes, weights), positive half.
const GL16_X: [f64; 8] = [
    0.095_012_509_837_637_44, 0.281_603_550_779_258_9,
    0.458_016_777_657_227_4, 0.617_876_244_402_643_8,
    0.755_404_408_355_003, 0.865_631_202_387_831_8,
    0.944_575_023_073_232_6, 0.989_400_934_991_649_9,
];
const GL16_W: [f64; 8] = [
    0.189_450_610_455_068_5, 0.182_603_415_044_923_6,
    0.169_156_519_395_002_5, 0.149_595_988_816_576_7,
    0.124_628_971_255_533_9, 0.095_158_511_682_492_78,
    0.062_253_523_938_647_89, 0.027_152_459_411_754_09,
];

/// log of the skew-normal lower-tail cdf F(y; a) = 2 int_{-inf}^y
/// phi(t) Phi(a t) dt for a > 0, y < 0, a y <= -3: the regime where
/// Phi(y) - 2 T(y, a) cancels. Log-concave integrand, so with
/// t = y - s and r = -(d/ds) log integrand at s = 0 the integrand is
/// below exp(-r s); integrate u = r s over [0, 60] on graded panels.
fn skew_normal_log_lower_tail(y: f64, a: f64) -> f64 {
    let g = |t: f64| -0.5 * t * t + log_ndtr(a * t); // log phi + const
    let g0 = g(y);
    let az = a * y;
    // phi(az)/Phi(az): inverse Mills ratio, from the stable log_ndtr
    let lam = (-0.5 * az * az - crate::LN_SQRT_2PI - log_ndtr(az)).exp();
    let r = -y + a * lam;
    let panels = [0.0, 0.5, 1.5, 3.5, 7.0, 13.0, 22.0, 36.0, 60.0];
    let mut acc = 0.0;
    for k in 0..panels.len() - 1 {
        let (u0, u1) = (panels[k], panels[k + 1]);
        let (c, h) = (0.5 * (u0 + u1), 0.5 * (u1 - u0));
        for j in 0..8 {
            for &sgn in &[-1.0, 1.0] {
                let u = c + sgn * h * GL16_X[j];
                let e = g(y - u / r) - g0;
                acc += GL16_W[j] * h * e.exp();
            }
        }
    }
    // F = 2 / sqrt(2 pi) * exp(g0) * acc / r
    std::f64::consts::LN_2 - crate::LN_SQRT_2PI + g0 + acc.ln() - r.ln()
}

/// log of the skew-normal survival S(x; alpha) = P(X > x) for the raw
/// (unstandardized) skew normal with shape alpha. For alpha >= 0 the
/// identity Phi(-x) + 2 T(x, alpha) adds two positive terms. For
/// alpha < 0, S(x; alpha) = F(-x; -alpha) by reflection, and in the
/// deep lower tail that cdf is integrated directly (#193).
pub fn skew_normal_log_sf(x: f64, alpha: f64) -> f64 {
    if alpha < 0.0 {
        let (y, a) = (-x, -alpha);
        if y < 0.0 && a * y <= -3.0 {
            return skew_normal_log_lower_tail(y, a);
        }
    }
    let sv = ndtr(-x) + 2.0 * owens_t::owens_t(x, alpha);
    sv.clamp(1e-300, 1.0).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_q_extremes() {
        // SciPy gammaincc references
        let cases = [
            (0.02, 6.0e-28, 0.711_307_731_585_341_1),
            (0.02, 30.0, 6.545_561_161_067_636e-17),
            (0.02, 100.0, 8.170_704_648_682_895e-48),
            (0.5, 1.0, 0.157_299_207_050_281_05),
            (2.0, 0.5, 0.909_795_989_568_950_1),
            (5.0, 100.0, 1.613_930_533_697_731_7e-37),
        ];
        for &(a, x, want) in &cases {
            let got = gamma_q(a, x);
            assert!((got / want - 1.0).abs() < 1e-12, "Q({a},{x}) = {got} vs {want}");
        }
    }

    #[test]
    fn skew_normal_negative_shape_tail() {
        // log S(1; -10) by adaptive quadrature of 2 phi(t) Phi(10 t) over
        // (-inf, -1] in the log domain: -58.591263384976216. SciPy's
        // skewnorm.sf gives 3.5821015037921536e-26 (log -58.5912627774),
        // itself 6e-7 relative off here; the identity Phi(-x) + 2T gives 0.
        let got = skew_normal_log_sf(1.0, -10.0);
        assert!((got + 58.591_263_384_976_216).abs() < 1e-12, "{got}");
        // continuity across the switch a*y = -3 and agreement with the
        // identity where it does not cancel
        for &(x, al) in &[(0.3, -10.0), (0.6, -5.0), (1.5, -2.0)] {
            let y = -x;
            let a = -al;
            let direct = (ndtr(-x) + 2.0 * owens_t::owens_t(x, al)).ln();
            let quad = skew_normal_log_lower_tail(y, a);
            assert!((direct - quad).abs() < 1e-9, "{x} {al}: {direct} {quad}");
        }
    }

    #[test]
    fn student_large_nu_is_normal_limit() {
        let s = student_sf(1.0, 1e50);
        assert!((s - ndtr(-1.0)).abs() < 1e-15);
    }
}
