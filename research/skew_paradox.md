# The skew paradox, resolved (July 2026)

HK finish-TIME noise residuals (75,013 runs, within-race standardized,
per-horse demeaned): skewness +19.6; slow tail 95th pct +3.33 sd vs fast
tail 5th pct -2.56 sd; mass beyond +2 sd is 7.6x the mass beyond -2 sd.
Times are violently right-skewed — bad days ARE much worse than good days
are good.

Yet skewed performance kernels are null for RANK prediction at every finish
position (skew_place_sweep.py, skew_deep_places.py). Three mechanisms:

1. Pairwise blindness: P(i beats j) depends on the difference of two iid
   noises, which is symmetric regardless of skew. Skew cancels in every
   head-to-head and only enters N>=3 rank behavior as higher-order
   corrections.
2. Monotone invariance: ranks are unchanged by any common monotone transform
   of performance. Skewness of observable times is compatible with symmetric
   noise under a curved clock; only skew surviving the best monotone
   straightening is rank-identifiable.
3. Tail censoring: ranks cap a catastrophic time at "finished last." The
   enormous slow tail that dominates time-skew is compressed by ranking into
   an outcome a symmetric kernel already prices for a longshot. Disaster
   mass matters for ranks only when catastrophe is FREQUENT and can strike
   anyone (F1 DNFs at ~25%: offday kernel helps every metric), not when it
   is rare and rank-censored (HK).

Practical rule for base_kernel: model catastrophe FREQUENCY (mixtures), not
time-domain shape (skew).
