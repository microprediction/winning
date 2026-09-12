"""python -m winning.ratings.verify [--profile P] [--only GLOB ...]
[--workers N] [--threads T] [--out DIR] [--seed-root S] [--quiet]

Caps BLAS/OpenMP threads BEFORE numpy is imported (the lattice kernels
are 1-D; BLAS threads only matter in the eigendecompositions of the
full-covariance path, and one thread per worker is the right shape).
Exit status: 0 clean (EXPECTED_APPROX allowed), 1 any FAIL, 2
UNDERPOWERED with no FAIL."""

import argparse
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m winning.ratings.verify")
    ap.add_argument("--profile", default="fast",
                    choices=("smoke", "fast", "full", "exhaustive"))
    ap.add_argument("--only", nargs="*", default=None,
                    help="glob(s) on check names")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1,
                    help="BLAS/OpenMP threads per process")
    ap.add_argument("--out", default=None, help="directory for the reports")
    ap.add_argument("--seed-root", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = str(args.threads)
    from .core import verify
    rep = verify(profile=args.profile, only=args.only, workers=args.workers,
                 verbose=not args.quiet, out=args.out, seed_root=args.seed_root)
    return rep.exit_code


if __name__ == "__main__":
    sys.exit(main())
