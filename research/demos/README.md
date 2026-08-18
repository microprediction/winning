# Demos

Explanatory scripts beside the papers, separated by which paper they
serve.

## siam2021/ — the original transform (SIAM J. Financial Math., 2021)

- `race_field_demo.py` — the multiplicative cavity: one shared survival
  field prices every competitor by division (plain NumPy, O(N) vs the
  naive O(N^2)).
- `cavity_downdate_demo.py` — the rank-one cavity twin: one inverse
  contains every leave-one-out inverse.
- Curated migrations of the thurstone examples land here (one demo per
  concept; no dumping).

## factor/ — the factor paper (Scalable Share Calibration for Factor
Multinomial Probit Models)

- `photo_finish_circuit.py` — the circuit: tie densities as
  conductances, a Newton step as an electrical solve, deletion flows
  against IIA (factor-similar runners gain disproportionately; +0.56
  mean correlation across 30 seeds). Interactive browser version at
  winning.microprediction.org/circuit.html, powered by `js/factor/`.
