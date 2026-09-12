"""Generative-model referee for the rating updates (the efficacy gate).

Promoted into the package on 2026-09-11: the samplers are
winning.ratings.simulate.sample_noise, the rejection posterior is
winning.ratings.verify.referee.mc_posterior, and the fixed
configurations (including the ones the bandits harness did not pick)
are winning.ratings.verify.referee.WINNER_CONFIGS / ORDER_CONFIGS. This
script runs the referee checks of the verifier so the ledger locator in
laplace_convolution_shortcut.md keeps pointing at a working command.

Run:  python research/adjudications/predictive_referee.py [--fast]
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from winning.ratings.simulate import sample_noise  # noqa: E402,F401
from winning.ratings.verify.referee import (WINNER_CONFIGS, ORDER_CONFIGS,  # noqa: E402,F401
                                            mc_posterior)


def main():
    from winning.ratings.verify import verify
    profile = "fast" if "--fast" in sys.argv else "full"
    rep = verify(profile=profile, only="referee.*")
    sys.exit(rep.exit_code)


if __name__ == "__main__":
    main()
