# Contributing to winning

Thanks for taking an interest. `winning` computes probabilities for races and contests (who wins,
finishing positions, and the inverse problem of recovering abilities), with ports in JavaScript, R,
Julia and Rust. Bug reports, questions and fixes are all welcome. This file says how to get help,
how to report a problem, and how to change the code.

Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting support

- **Questions about using winning:** open a
  [GitHub issue](https://github.com/microprediction/winning/issues). Questions are welcome there;
  you do not need to have found a bug. The [documentation](https://winning.microprediction.org)
  and the README examples are the first place to look.
- **Who answers:** the maintainer, Peter Cotton, reads every issue. Support is best effort: there
  is no guaranteed response time, but bugs that give silently wrong probabilities are treated as
  the top priority.
- **Security problems:** please don't open a public issue. See [SECURITY.md](SECURITY.md).
- **Conduct reports and other private matters:** use the [private reporting form](https://github.com/microprediction/winning/security/advisories/new) on the Security tab, which reaches only the maintainer.

## Governance

winning is maintained by one person, Peter Cotton, who reviews and merges pull requests and makes
releases for every language. Decisions about scope and design are discussed in the open, on issues
and pull requests, before they are made. The Python package is the reference: the other ports must
agree with it, and a change to the reference is not finished until the ports either follow it or
the divergence is recorded. Anyone whose contributions are substantial and sustained can ask to
become a co-maintainer.

## Reporting a bug

Open an issue and include:

1. what you ran: a short, self-contained script, ideally with a fixed random seed;
2. what you expected, and what happened instead (the full traceback, or the probabilities that are
   wrong);
3. versions: `python -c "import winning, numpy, sys; print(winning.__version__, numpy.__version__, sys.version)"`
   (or the equivalent for the R, Julia, JavaScript or Rust port), and your operating system.

A probability that is wrong but raises no error is a bug, and the most important kind. Please
report it even if you are not sure.

## Proposing a change

- **Small fixes** (typos, documentation, an obvious bug): open a pull request directly.
- **New methods or changes to the public API:** open an issue first, so the design can be agreed
  before you write the code. Say how the result can be checked: a closed form, a reference
  implementation, a Monte Carlo comparison or an invariance.
- Every pull request should target `main`, keep to one topic, add or update tests, and pass CI.
- A change to the Python reference that the ports implement should either update the ports in the
  same pull request or say which ports still need it. `parity/` holds the cross-language checks.

## Development setup

You need Python 3.10 or later and git.

```bash
git clone https://github.com/microprediction/winning.git
cd winning
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -e ".[test]"
pytest tests                       # the core suite
ruff check winning                 # lint
```

The ports live in `js/`, `r/`, `julia/` and `rust/`, each with its own tests. Coding agents
working in this repository should read [CLAUDE.md](CLAUDE.md) first.

## Releasing (maintainer)

Publishing a GitHub release runs `.github/workflows/publish.yml`, which runs the full suite, test-
installs the built wheel in a clean environment, and uploads to PyPI by trusted publishing (no
stored PyPI token). The compiled extensions publish through `publish-fastrace.yml` and
`publish-fastmvn.yml` the same way, and Julia releases are registered through TagBot.
