# Security policy

## Reporting a vulnerability

Please report security problems privately, not in a public issue:

- through GitHub's private vulnerability reporting: the
  [Report a vulnerability](https://github.com/microprediction/winning/security/advisories/new)
  button on this repository's Security tab; or
- by email to peter.cotton@microprediction.com.

Include what you found, how to reproduce it, and which package and version it affects (the Python
package on PyPI, or the JavaScript, R, Julia or Rust port).

## What to expect

The maintainer will acknowledge a report within a week, keep you informed while it is fixed, and
credit you in the advisory unless you prefer otherwise. Fixes are released as new versions, with a
GitHub security advisory describing the problem and the affected versions.

## Supported versions

Security fixes go into the latest release of each package. Older releases are not patched.

## Scope

winning is a numerical library. The main risks are to the supply chain: a compromised release,
dependency or build workflow. Releases are published from GitHub Actions by PyPI trusted
publishing, with no long-lived publishing token stored in the repository.
