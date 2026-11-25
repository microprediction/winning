# Reproducing this paper

The paper's numbers were produced at repository tag **`paper-r12`**
(package version 1.4.0 plus the reproduction fixes). To regenerate them:

```sh
git checkout paper-r12
cd docs/latex_src/general_inversion
make env && make          # accuracy claims, ~15 minutes
make full                 # long runs and the R referees
make timings              # wall clock, on a quiet machine
```

At that tag the paper's directory is `docs/latex_src/general_inversion/`,
where `REPRODUCE.md` gives the pinned environment, the determinism notes
and the known limits, and `CLAIMS.md` maps every claim to its `make`
target. The scripts in this directory and under `research/` on `main`
track the current package and can differ from the tag in the last digits.

`MANIFEST.md` in this directory records the provenance of the paper's
tables and numbers: command, seed and log line where a committed command
produces them, and a list of the measurements that none regenerates. The analyses added in the journal
revision (`selfconv.py` here, `research/general_sigma/paired_ensembles4.py`,
`research/inversion_ablation/run_convergence_battery.py`) run against the
`paper-r12` package; the branch `paper/inversion-r13` carries them on
top of that tag.
