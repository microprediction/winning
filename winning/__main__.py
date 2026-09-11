"""``python -m winning`` and the ``winning`` console script: print the
installed version and run the package's own contract (goldens, identities,
round trips), so an install is checked in one command. Exit status 1 when
the contract fails."""

import sys


def main(argv=None):
    import winning
    from winning import contract

    print(f"winning {winning.__version__} ({winning.__file__})")
    ok = bool(contract.verify(verbose=True))
    print("contract:", "verified" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
