# Demos

Explanatory scripts for the factor calibration core, beside the paper
in `papers/factor-probit-transform/`.

- `race_field_demo.py` — the multiplicative cavity: one shared survival
  field prices every competitor by division (the paper's Algorithm 1 in
  miniature, plain NumPy).
- `cavity_downdate_demo.py` — the rank-one cavity twin: one inverse
  contains every leave-one-out inverse.
- `photo_finish_circuit.py` — the circuit demo: tie densities as
  conductances, a Newton step as an electrical solve, and deletion
  flows against IIA (factor-similar runners gain disproportionately;
  +0.56 mean correlation across 30 seeds). Interactive browser version
  at winning.microprediction.org/circuit.html, powered by `js/factor/`.

The Laplacian Newton-CG demonstration against the thurstone API lives
with that package (`thurstone/examples/laplacian_newton_demo.py`);
experiment 23 here covers the same ground through `winning.factor`.
