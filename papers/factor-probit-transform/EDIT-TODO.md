# Deferred edits (from author-rewrite session)

- Vasicek citation: add as the CREDIT-RISK instance in the Related Work
  "factor dimension reduction is old" paragraph (line ~1220), NOT as
  priority over Butler (Vasicek 1987/1991/2002 all post-date
  Butler-Moffitt 1982). Frame: one-factor Gaussian conditional-
  independence integral -> Basel II IRB capital, adopted because it beat
  Monte Carlo. Reinforces the intro's "each field reinvents factor
  integration" thesis + ties to the trillions/market-cap theme.
  Ref: O.A. Vasicek, "Loan portfolio value," Risk 15 (2002) 160-162.
- Racing citation: "second-place probabilities better modeled by probit
  than logit" -> \citep{henery1981,lo1994} (both already in bib).
- China/US line in intro: author to decide (flagged as venue-tone risk,
  not logic error).

## From the 2026-08-16 full referee review (deferred; need Peter's call)
- Title: DECIDED 2026-08-16 — stays "Scalable Probit Share Calibration".
- Intro color cuts (China/US, indexes, biology, second-place horses,
  "built into every neural network", "universal for one reason"): the
  reviewer wants nearly all of it gone. Peter's prose; his call.
- Full 10-section restructure proposal (algorithmic spine first, all
  interpretations moved to a late Connections section). Conflicts with
  Peter's own 6-section structure from this week.
- Extend exp24 per-alternative RQMC comparison: several N, k=2..4,
  share buckets, heteroskedastic designs; move from Related Work to §4.
- Experiment numbers out of prose into an appendix table (reviewer);
  conflicts with the committed-script ethos in the prose.
- Ordered-probit/Lazear-Rosen "same model" looseness in intro ¶1.
- D-range reporting for the original exp16 robustness suite (new exp33
  covers ratios 1e2/1e3; retrofitting old suite optional).
- Factorial replication: DONE 2026-08-17 (experiment 36, 20 seeds,
  factor probit wins 20/20 in every stratum). Remaining caveat: single
  truth family.
- Optional: 1000-problem randomized stress suite for empirical global
  convergence (failure count, worst iterations vs spectral gap).
- Real-data inner-inversion demo (reviewer suggestion): take V from
  actual product characteristics or an existing fitted factor MNP and
  demonstrate ONLY the inner inversion/removal problem. Removes the
  "synthetic algorithm looking for a use case" objection without
  expanding claims.
- Reviewer prefers Further structure compressed to a paragraph with the
  transport/trace material in an appendix (currently its own section
  after the numerics). Editorial; Peter's call.
- Intro trim: reviewer asks again for 40-50% cut of pp. 1-2 color.
