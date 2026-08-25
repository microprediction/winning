# Deferred edits (from author-rewrite session)

- Vasicek citation: DONE 2026-08-24 — credit-risk sentence added to the
  Related Work dimension-reduction paragraph, after Dunnett, no priority
  claim over Butler.
- Racing citation: DONE 2026-08-24 — henery1981/lo1994 already attached
  to the second-place sentence in Potential applications.
- Prior-art citations from papers/prior-art-inversion-and-shared-field.md:
  DONE 2026-08-24 — Li 2018 (complement framing), Lambert 1975 (earliest
  shared-field), Chiang 1961 (deletion ensemble), Anderson-Ghurye 1977 +
  Mukherjea-Stephens 1990 (identification), Thurstone 1945 + Guilford
  1937 (psychometric forward integral / backward problem). Two bibitem
  titles flagged % VERIFY (Chiang, Mukherjea-Stephens: titles from
  memory).
- Intro rewrite: DONE 2026-08-24 — leads with the problem then the
  hardness record (Torgerson, GHK, Conlon, PyBLP, BLP 2004, Freyberger,
  Sawtooth, McFadden-Train); panorama moved to Potential applications.
  Three quote sets flagged % VERIFY pending the verification agent.
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
