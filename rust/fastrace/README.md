# fastrace: Rust kernel for the factor-probit share transform

Fused, rayon-parallel implementation of the min-wins factor race
(the `thurstone`/`raceutil` shared-survival-field pass). Log-domain
throughout (log_ndtr via libm::erfc with asymptotic tail), x-tiled for
cache, parallel over factor nodes.

Measured (Apple M4, vs single-threaded NumPy reference, GH order 15, k=2,
L=1501; agreement 8e-17):

| N | NumPy | Rust | speedup |
|---|---|---|---|
| 1000 | 4.24 s | 0.71 s | 6.0x |
| 5000 | 21.6 s | 3.9 s | 5.5x |

Build: `pip install maturin && maturin develop --release` (needs Rust
toolchain). Exposes `fastrace.win_probabilities_factor(mu, V, D, F, W,
points=1501) -> (p, total)`.

Also implemented and measured (quiet machine, medians of 3):

- forward_and_slopes: the calibration pass; drop-in solver `rustcal.py`
  calibrates N=1000 in 8s, N=5000 in 45s (identical iterations/residuals).
- jacobian_vector_product (ibp + grid forms): parity 2e-14, 5.5x.
- win_probabilities_factor_separated (Chebyshev low-rank pass):

| N | exact numpy | rust exact | rust separated r=384 | err |
|---|---|---|---|---|
| 1000 | 4.12 s | 703 ms (6x) | 34 ms (121x) | 2.7e-5 |
| 5000 | 21.3 s | 3.78 s (6x) | 83 ms (255x) | 6.1e-5 |

Not yet ported: deletion ensemble; separated-pass slopes (would take
calibration to ~1s at N=5000); SIMD transcendentals (~4x further).
