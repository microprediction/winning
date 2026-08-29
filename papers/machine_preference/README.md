# Choice-Set Restriction in Machines and People

**Status: draft, no venue.** First version November 2024; this version
15 August 2026. Not submitted anywhere.

## What is here

`paper.tex` / `paper.pdf` only. The paper uses LLMs as a direct-readout
instrument for choice distributions and compares the IIA discount they
exhibit (lambda ~ 0.69) against human estimates (lambda ~ 0.42 / 0.19)
and a Thurstonian benchmark.

## Where the supporting material lives

The manuscript previously referred to "the appendix of the working
version" and to figure sources that "live with the original draft" --
neither of which is in this directory. Both point at material that IS
in this repository, under a different name:

- **The ninety-nine category/adjective combinations**:
  `research/polysemy_pilot/qvols/*.yaml` (99 categories, each with an
  adjective list and a `prompt_pair_template`).
- **The sweep that consumes them**: `research/polysemy_pilot/sweep.py`
  (question type x adjective x model), with raw logs in
  `research/polysemy_pilot/sweep_raw.jsonl`.

The paper now cites those paths directly instead of an absent appendix.

## Open

- **Figures 1 and 2 of the July 2024 version** (entropy scatter;
  per-category RMSE bars) were never carried into this repository. They
  are regenerable from the sweep above, but the plotting sources are
  gone. The relevant `\section` carries a comment saying so.
- No venue has been chosen.
