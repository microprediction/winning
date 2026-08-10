"""Model tiers for the exact-probability batteries.

HEADLINE carries the pooled statistics reported in the paper; it is frozen so
those numbers stay comparable across runs. BREADTH adds cheaper models used to
widen coverage (more categories, greater deletion depth, replications) and to
extend the capability ladder downward. Pooled results are always reported for
HEADLINE separately from ALL, since pooling across tiers of very different
quality would move the headline number for reasons that have nothing to do
with the choice law.

All five return full 20-entry top_logprobs at ~29-39 prompt tokens. The GPT-5
tier is unusable here: it refuses logprobs outright (403, "You are not allowed
to request logprobs from this model"), so exact measurement is confined to the
4.x tier.
"""

HEADLINE = ["gpt-4o-mini", "gpt-4o", "gpt-4.1"]
BREADTH = ["gpt-4.1-mini", "gpt-4.1-nano"]
ALL = HEADLINE + BREADTH
