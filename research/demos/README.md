# Demos

Explanatory scripts for the factor calibration core, beside the paper
in `papers/factor-probit-transform/`.

- `race_field_demo.py` — the multiplicative cavity: one shared survival
  field prices every competitor by division (the paper's Algorithm 1 in
  miniature, plain NumPy).
- `cavity_downdate_demo.py` — the rank-one cavity twin: one inverse
  contains every leave-one-out inverse.
- Planned: the photo-finish circuit demo (tie densities as
  conductances, Newton steps as electrical solves, deletion flows vs
  IIA), in Python (figure) and in the browser via `js/factor/`.

The Laplacian Newton-CG demonstration against the thurstone API lives
with that package (`thurstone/examples/laplacian_newton_demo.py`);
experiment 23 here covers the same ground through `winning.factor`.
