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
