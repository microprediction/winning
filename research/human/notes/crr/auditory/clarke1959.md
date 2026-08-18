# Clarke (1959) — percent correct vs number of alternatives: threshold vs CRR vs signal detectability

## Citation

Clarke, F. R. (1959). Proportion of Correct Responses as a Function of the Number of
Stimulus-Response Alternatives. *Journal of the Acoustical Society of America*, **31**(6_Supplement),
835. doi:10.1121/1.1930396

This is a **one-page meeting abstract** in the ASA meeting supplement, not a full paper. Work
sponsored by the Operational Applications Laboratory, Air Force Cambridge Research Center, and the
U.S. Army Signal Corps Engineering Laboratories, Fort Monmouth. (Note the AFCRC sponsorship: the same
laboratory that employed Pollack and Decker.)

## Stimuli and master response set

Four families of signal ensembles, verbatim from the abstract: "*(a) speech signals; (b) a sinusoidal
signal occurring in one of n intervals in time; (c) sinusoidal signals of varying amplitude; and (d)
sinusoidal signals of varying frequency.*"

Closed-set identification with full knowledge of the alternatives: "*In all of these experiments the
listeners had complete knowledge of the set of stimuli under test and limited their responses
accordingly.*"

## Restricted response sets (state whether nested, overlapping, or a relabelling)

**Nested — n is the independent variable.** The whole experiment is a sweep over the number of
stimulus-response alternatives. But **no confusion matrices are reported**: the dependent variable is
the scalar proportion of correct responses as a function of n.

## What numbers are printed (master matrix? restricted matrices? counts or proportions? which table numbers?)

**Nothing but percent correct, and in a one-page abstract not even that in tabulated form.** No master
matrix, no submatrices, no tables. This is the paper that the task brief anticipates: it manipulates
response-set size but publishes only percent correct, so it is unusable as a data source.

Its value is entirely conceptual, and it is considerable — see the final section.

## Access (a DIRECT url you have fetched, plus whether it is open, paywalled or Wayback-only)

- `https://api.crossref.org/works/10.1121/1.1930396` — FETCHED 200. **Open**, and because the item is
  a meeting abstract, the Crossref abstract field *is* the complete published content. Full text
  quoted below.
- `https://api.unpaywall.org/v2/10.1121/1.1930396?email=...` — FETCHED 200. `is_oa: true`,
  `oa_status: "bronze"`, one publisher OA location:
  `https://pubs.aip.org/asa/jasa/article-pdf/31/6_Supplement/835/12009457/835_2_online.pdf`
  So AIP considers this item **free to read**.
- `https://pubs.aip.org/asa/jasa/article-pdf/31/6_Supplement/835/12009457/835_2_online.pdf` —
  FETCHED via curl (**403**, Cloudflare) and via WebFetch (**403**). Free in principle, but AIP's bot
  protection blocks this environment. **A human browser should get it without a subscription.**
- Wayback CDX for `pubs.aip.org/asa/jasa/article-pdf/31/6_Supplement/835*` and
  `.../article/31/6_Supplement/835*` — FETCHED, **no snapshots**. (Neighbouring pages 830-832 in the
  same supplement do have snapshots, mostly 403 captures.)

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**UNUSABLE as data — one-page meeting abstract, percent correct only, no confusion matrices.**

Worth ten minutes of a human browser's time anyway, because the abstract is free at AIP and because
of what it says (below). Do not spend effort digitizing it.

## What the authors concluded about CRR, quoted verbatim where possible

The complete published text, verbatim:

> "The effect of the number of stimulus-response alternatives on the human observer's probability of
> making a correct response was predicted by three models: a simple threshold model; the constant-ratio
> rule with simplifying assumptions; and the theory of signal detectability with simplifying
> assumptions. Four types of signal ensembles were used in these experiments: (a) speech signals; (b) a
> sinusoidal signal occurring in one of n intervals in time; (c) sinusoidal signals of varying
> amplitude; and (d) sinusoidal signals of varying frequency. In all of these experiments the listeners
> had complete knowledge of the set of stimuli under test and limited their responses accordingly. The
> simple threshold model failed to account for any of the data. The simplified version of the
> constant-ratio rule and the simplified version of the theory of signal detectability were both
> compatible with the data obtained in the speech experiments. Also, over the small range tested, both
> handled data obtained when a sinusoidal signal occurred in one of n intervals. No model tested was
> sufficiently complex to account for data when the sinusoidal signals varied only in amplitude or
> only in frequency. (Work under sponsorship of Operational Applications Laboratory, Air Force
> Cambridge Research Center, and U. S. Army, Signal Corps Engineering Laboratories, Fort Monmouth.)"

**Why this abstract matters out of all proportion to its length.** It is a 1959 head-to-head contest
between exactly the two families the present project pits against each other — "the constant-ratio
rule" (Luce / Gumbel / proportional renormalization) and "the theory of signal detectability"
(Gaussian / Thurstone) — plus a threshold model as a straw man. Clarke's verdict is that on percent
correct the two are **indistinguishable** on speech ("both compatible with the data obtained in the
speech experiments") and **both fail** on unidimensional tone continua ("No model tested was
sufficiently complex to account for data when the sinusoidal signals varied only in amplitude or only
in frequency").

That is the strongest possible historical argument for the project's framing: percent correct cannot
separate Gaussian from Gumbel, so the literature settled for a scalar that had no power, and the
confusion matrices that *would* have separated them were never used for that purpose. The
distributional forecasting contest was available in 1959 and was not run.

Clarke also privately abandoned the CRR for unidimensional stimuli on the strength of this work.
Hodge (1967, *Percept. Psychophys.* 2, p. 429), verbatim:

> "In an experiment with simple auditory stimuli, e.g., tones varying in frequency or intensity,
> Clarke (1959) noted that the rule tended to fail. In Clarke's opinion, [footnote 2: F. R. Clarke,
> personal communication, 1960] the rule failed because the ordering inherent in single dimensional
> stimuli produces contextual constraints or biases which, by definition, are incompatible with the
> CRR."

Lee (1968, *Percept. Psychophys.* 4, p. 219), verbatim:

> "Clarke later put forward doubts on the adequacy of the CRR for unidimensional stimuli, but these
> doubts were based on empirical work."
