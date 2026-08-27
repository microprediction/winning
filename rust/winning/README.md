# winning

Lattice race kernels, pure rust: the compute core of the
[`winning`](https://github.com/microprediction/winning) ecosystem
(python package `winning`, R package `winning`, pyo3 wheel `fastrace`).

Given contestants whose performance is `mu + noise` (min wins), computes
all `N` win probabilities in one shared-survival-field lattice pass, for
noise covariance described by any of the race grammars:

- **factor**: `Sigma = V V' + diag(D)` — `forward_kernel` (probabilities
  and inversion slopes), `jvp_kernel`, Chebyshev-separated variant
- **blocks**: block-diagonal rank-1 or rank-r plus diagonal —
  `block_kernel` (hybrid in-memory/streaming), `block_kernel_r`
- **tree**: hierarchy of uniform shared effects — `tree_kernel`
  (two message passes)
- **classic**: the original state-price lattice calibration of
  Cotton (2021), *SIAM J. Financial Math.*, dead heats handled exactly —
  `classic_state_prices`, `classic_calibrate`
- **GHK** all-shares for benchmark comparisons

Parallel over factor nodes (rayon), log-domain throughout, x-tiled for
cache. The python reference implementation is the spec; the language
ports are parity-locked to it (see `parity/` in the repository).
