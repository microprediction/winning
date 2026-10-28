# Fishing regression bisection (2026-10-13)

The 2026-10-13 Fishing rerun (../2026-10-13/) gave python exact referee
logLik -1215.86 against the recorded -1212.82 (BENCHMARK.md, 2026-09-06),
and the julia exact fit crashed. Bisected over main's first-parent
history from 8e1815f4 (the commit that recorded the table) with
`fish_at.sh` (python fit + julia exact fit at a SHA, checked out in a
separate worktree; same venv: python 3.14.6, numpy 2.5.1, scipy 1.18.0,
julia 1.12.6; nice 19, 2 threads). `fish_bisect.py` and `bisect2.py`
drive it; index i is the i-th first-parent commit after 8e1815f4.
(The last index, 29acbdc6, is a bench-oct13 results commit whose code
is origin/main a7f7d9fd unchanged.)

Raw lines (index, sha | python | julia exact):

    0 8e1815f4 | python exact: referee logLik -1212.82 +- 0.10 in 9.1s | julia exact logLik -1214.52 12.8s conv true
    98 15a396e8 | -1212.82 +- 0.10 in 9.0s | julia -1214.52 12.7s
    110 138e7adb | -1212.82 +- 0.10 in 9.3s | julia -1214.52 12.9s
    116 262028e4 | -1212.82 +- 0.10 in 9.4s | julia -1214.52 12.7s
    119 279c4f91 | -1212.82 +- 0.10 in 9.2s | julia -1214.52 12.8s
    120 b3b55b29 | -1212.82 +- 0.10 in 9.1s | julia -1214.52 12.9s
    121 34a9b39c (#291) | -1215.24 +- 0.08 in 9.8s | julia -1214.52 24.5s
    122 7b02458e | -1215.24 +- 0.08 in 9.9s | julia -1214.52 24.4s
    147 040d6e46 | -1215.24 +- 0.08 in 9.6s | julia -1214.52 24.6s
    171 643cd797 | -1215.24 +- 0.08 in 9.7s | julia -1214.52 24.6s
    177 2cc78f89 | -1215.24 +- 0.08 in 9.6s | julia -1214.52 24.4s
    180 bba13fd6 (#689) | -1215.24 +- 0.08 in 9.6s | julia -1214.52 24.4s
    181 272a1c7b (#688) | -1215.24 +- 0.08 in 9.6s | julia -1214.52 24.4s
    182 216700b5 (#691) | -1215.24 +- 0.08 in 9.6s | julia -1214.52 24.4s
    183 4b43e9ca (#693) | -1215.86 +- 0.12 in 21.0s | ERROR: DomainError with -196.19209060821765 (sqrt, MultinomialProbit.jl:568)
    196 29acbdc6 (= a7f7d9fd code) | -1215.86 +- 0.12 in 21.3s | ERROR: DomainError with -196.19209060821765

First bad commits:
- #291 (34a9b39c, 2026-09-25, "Stop the MNP tail likelihood going flat"): python optimum -1212.82 -> -1215.24 (-2.4 nats); julia exact fit time 12.7 s -> 24.5 s (same julia optimum).
- #693 (4b43e9ca, 2026-10-13, fix3-likelihood-ratings): python -1215.24 -> -1215.86 and python fit time 9.6 s -> 21.0 s (2.2x); julia exact fit crashes, sqrt of a negative tilt curvature h = -196.2 in _choice_terms (the tilt comment says log h has curvature <= -1, so h should be >= 1).

The referee is not what moved: `ref_cross.py` scores every fitted theta
with the referee code of both 8e1815f4 and a7f7d9fd:

    referee code at 8e1815f4:  8e1815f4 -1212.82 | b3b55b29 -1212.82 | 216700b5 -1215.24 | a7f7d9fd -1216.09
    referee code at a7f7d9fd:  8e1815f4 -1212.82 | b3b55b29 -1212.82 | 216700b5 -1215.24 | a7f7d9fd -1216.46

So the fitted point is genuinely ~3 nats worse under the shared referee.
