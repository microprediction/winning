# Holloway (1971) — a test with power finds the dependency the CRR literature had denied

## Citation

Holloway, C. M. (1971). A Test of the Independence of Linguistic Dimensions. *Language and Speech*,
**14**(4), 326-340. doi:10.1177/002383097101400402
Medical Research Council Applied Psychology Unit, Cambridge.

Third and last of Holloway's three papers. Reanalysis, no new data collection.

## Stimuli and master response set

**No new experiment.** Two datasets are reanalysed, both consonants-in-noise:

1. **Holloway's own experiment**, i.e. the data of Holloway (1970), *QJEP* 22, 467-474 (consonant
   recognition at two levels of decision complexity — see `holloway1970.md`). Verbatim: "*A new
   procedure for data analysis is applied to the results of an experiment reported by the author.*"
2. **Miller & Nicely (1955)**, *JASA* 27, 338-352, doi:10.1121/1.1907526 — the sixteen English
   consonants spoken over distorted and noise-masked channels. Verbatim: "*Application of the present
   technique to the data of Miller and Nicely (1955) also shows a significant dependency effect.*"

Miller & Nicely's master set is the sixteen consonants /p, t, k, f, θ, s, ʃ, b, d, g, v, ð, z, ʒ, m, n/,
crossed with five articulatory features (voicing, nasality, affrication, duration, place of
articulation).

Holloway's full reference list (9 items), retrieved from OpenAlex, is the tightest statement of what he
took himself to be refuting — it is almost exactly the auditory CRR spine:

- Miller & Nicely (1955), *JASA* 27, 338-352
- Clarke (1957), *JASA* 29, 715-720
- Clarke & Anderson (1957), *JASA* 29, 1318-1320
- Clarke (1959), *JASA* 31, 835
- Pollack & Decker (1960a), "Consonant Confusions and the Constant Ratio Rule", *L&S* 3, 1-6
- Pollack & Decker (1960b), "Perception of Consonant Voicing in Noise", *L&S* 3, 155-163
- Holloway (1968), *QJEP* 20, 336-350
- Holloway (1970), *QJEP* 22, 467-474
- Peterson & Barney-era consonant study: "Study of Twenty-Six Intervocalic Consonants as Spoken and
  Recognized by Four Language Groups" (1966), *JASA*, doi:10.1121/1.1909899

Nine references, six of them Clarke/Pollack CRR papers. This is a paper written to overturn that
corpus.

## Restricted response sets (state whether nested, overlapping, or a relabelling)

**Inherited, and of two different kinds.**

From Holloway (1970): **nested** restrictions (two levels of decision complexity, sets chosen to fix
the number of relevant dimensions).

From Miller & Nicely (1955): the relevant structure is a **relabelling / factorisation**, not a
pruning. Miller & Nicely's 16x16 matrices are analysed by decomposing the response into five feature
decisions; Holloway's "test of the independence of linguistic dimensions" asks whether the joint
16x16 cell probabilities factorise into the product of the marginal feature-wise probabilities. That is
an aggregation/independence test on a fixed response set, not a subset-prediction test.

**Both kinds bear on IIA, and Holloway rejects independence in both.** The connection to the CRR is the
one Holloway's own citation list makes: the CRR literature had used feature independence as the
mechanism explaining why proportional renormalization should work on speech. Remove independence and
the mechanism goes.

## What numbers are printed (master matrix? restricted matrices? counts or proportions? which table numbers?)

NOT VERIFIED (paywalled, 15 pages). Assessment:

- **Test statistics and their significance are certainly printed** — the abstract reports a "*small but
  reliable dependency effect*" and a "*significant dependency effect*" for Miller & Nicely, so there
  are numbers with p-values attached. This is the *second* paper in the branch, after Morgan (1974), to
  attach a significance test to an independence/IIA claim about speech confusions. Note the two are
  contemporaneous and, on the evidence of their reference lists, unaware of each other: Morgan (1974)
  does not cite Holloway, and Holloway (1971) does not cite Morgan.
- **Whether the underlying matrices are re-printed is unknown.** For the Miller & Nicely part it does
  not matter: **those matrices are the most widely reproduced confusion data in existence and are
  freely available elsewhere** (a re-typeset copy,
  `50YearsLate-RepeatingMillerNicely55.06.pdf`, from the Allen group at Illinois, was already present in
  this session's working directory; the source directory
  `https://jontalle.web.engr.illinois.edu/Public/` was fetched successfully, HTTP 200). So the Miller &
  Nicely reanalysis is reproducible from open data even without buying Holloway.
- For the Holloway (1970) part it matters a great deal, and it is unresolved — see `holloway1970.md`.
- Fifteen pages of *Language and Speech* is ample room for matrices plus a methods section.
- Table numbers unknown.

## Access (a DIRECT url you have fetched, plus whether it is open, paywalled or Wayback-only)

- `https://api.crossref.org/works/10.1177/002383097101400402` — FETCHED 200. **Open.** Full author
  abstract (quoted below), affiliation, volume 14, pages 326-340.
- `https://api.unpaywall.org/v2/10.1177/002383097101400402?email=...` — FETCHED 200. `is_oa: false`,
  `oa_status: "closed"`, zero OA locations.
- `https://api.openalex.org/works/https://doi.org/10.1177/002383097101400402` — FETCHED 200. Full
  9-item reference list retrieved (enumerated above).
- `https://api.semanticscholar.org/graph/v1/paper/DOI:10.1177/002383097101400402` — FETCHED 200,
  `openAccessPdf.status: "CLOSED"`, abstract elided by publisher.
- Sage landing page under `https://journals.sagepub.com/doi/10.1177/002383097101400402` — Sage returns
  Cloudflare **HTTP 403** to curl and WebFetch from this environment (verified on the sibling
  *Language and Speech* DOI 10.1177/002383096000300101). **Paywalled.**
- Wayback CDX for `*sagepub.com*002383097101400402*` — FETCHED (one query timed out at 40 s, retried),
  **no snapshots**. Not even Wayback-only.
- Freely available substitute for one of its two datasets:
  `https://api.crossref.org/works/10.1121/1.1907526` — FETCHED 200, complete Miller & Nicely (1955)
  abstract; and the re-typeset matrices are open at the Illinois directory above.

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**NEEDS LIBRARY ACCESS for the method and the Holloway-1970 result; the Miller & Nicely half is USABLE
NOW from open sources.**

Split the job:
- *Miller & Nicely reanalysis*: do not buy Holloway for this. The matrices are open. Recompute the
  independence test directly, and compare a Thurstonian and a Luce fit on the same 16x16 data. The
  historical claim ("a significant dependency effect") is already quotable from the open abstract.
- *Holloway's own 1970 data*: needs the 1970 paper, not this one.
- *The paper itself*: worth ordering as the third item in a single Holloway request, because it supplies
  the verbatim conclusion that the independence assumption underpinning the auditory CRR fails, from an
  author who had himself endorsed independence three years earlier.

Priority: medium. High citation value, low unique data value.

## What the authors concluded about CRR, quoted verbatim where possible

Holloway's abstract, verbatim and in full:

> "Several investigations of the perception of consonants spoken in noise have purported to show the
> independence of the linguistic dimensions which define a consonant. A new procedure for data analysis is
> applied to the results of an experiment reported by the author. This analysis suggests that there is a
> small but reliable dependency effect. Application of the present technique to the data of Miller and
> Nicely (1955) also shows a significant dependency effect."

Three things worth extracting from that single paragraph.

**One.** "*have purported to show*" is a direct challenge to the corpus, and the corpus is named in his
reference list: Clarke (1957, 1959), Clarke & Anderson (1957), Pollack & Decker (1960a, 1960b), Miller &
Nicely (1955).

**Two.** "*small but reliable*" is the exact phrase the whole 1957-1970 literature could not produce,
because it had no test. Clarke's criterion was an eyeball .10 on proportions (Hodge 1967, p. 431:
"*the percentage of the absolute differences which exceed a difference of .10, a criterion value
suggested by Clarke (1957)*"); Pollack & Decker's was a 4-percentage-point average deviation; Hodge
(1967, p. 430) states flatly that "*a satisfactory statistical test is not available (Clarke, 1957)*".
Once a test with power arrives — Holloway's in 1971, Morgan's in 1974 — the answer flips.

**Three.** The dependency is *small*. Holloway is not claiming the CRR is grossly wrong; he is claiming
it is reliably wrong. That is precisely the regime in which a scoring rule on a distributional forecast
beats a hypothesis test, and it is the strongest argument available for reframing this literature as a
forecasting contest rather than an accept/reject exercise.

What the original authors had concluded, for contrast — verbatim from Miller & Nicely (1955), whose
abstract is open:

> "The indications are that the perception of any one of these five features is relatively independent of
> the perception of the others, so that it is as if five separate, simple channels were involved rather
> than a single complex channel."

Holloway (1971) is the refutation of that sentence.
