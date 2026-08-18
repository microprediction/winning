# Holloway (1970) — consonant recognition at two levels of decision complexity: message sets chosen to control dimension count

## Citation

Holloway, C. M. (1970). Consonant Recognition with two Levels of Decision Complexity. *Quarterly
Journal of Experimental Psychology*, **22**(3), 467-474. doi:10.1080/14640747008401921
PMID 5470326. Medical Research Council's Applied Psychology Unit, Cambridge.

Second of Holloway's three papers (1968, 1970, 1971). This is the one with the response-set-size
manipulation.

## Stimuli and master response set

**Consonants in noise.** The design's distinguishing feature is that the message sets were not chosen
by convenience or by size alone, but **selected so as to fix the number of linguistic dimensions the
listener must discriminate**. Verbatim from the abstract:

> "In the present experiments the particular items appearing in a message set have been selected in
> order to control precisely the number of dimensions relevant to the discrimination task, and
> performance is estimated in terms of the discrimination of each relevant dimension."

The target being tested is the **perceptual tuning hypothesis**, verbatim:

> "The perceptual tuning hypothesis can be expressed as a prediction that discrimination of a particular
> stimulus dimension will be enhanced if it is the only dimension requiring discrimination and not one of
> a set of dimensions in a multidimensional discrimination task."

and the criticism of the prior literature, verbatim:

> "In previous tests of the tuning hypothesis, using speech material, the size of a message set has been
> manipulated as an indirect means of controlling the number of dimensions an observer is required to
> discriminate."

That sentence is aimed squarely at the CRR literature. Holloway's full reference list (17 items),
retrieved from OpenAlex, includes Clarke (1957), Clarke & Anderson (1957), Pollack & Decker (1960a),
Miller & Nicely (1955), Miller (1954) "Accuracy of recognition with alternatives before and after the
stimulus", Pollack's *Message Uncertainty and Message Reception* I (1959) and II (1960) and III (1963),
Garner (1962) *Uncertainty and Structure*, Jakobson/Fant/Halle (1952) *Preliminaries to Speech
Analysis*, Luce (1963) Handbook chapter, and **Green, Birdsall & Macnee (1958) "The effect of vocabulary
size on articulation score"** (green OA at Michigan Deep Blue — see
`green_birdsall_macnee1958.md`).

Exact consonant inventories and set sizes: NOT recoverable from the abstract.

## Restricted response sets (state whether nested, overlapping, or a relabelling)

**Nested, at exactly two levels — "two levels of decision complexity" is the title.** Sets of two
different sizes/dimensionalities over a common consonant vocabulary, with the smaller sets chosen so
that fewer linguistic dimensions are relevant.

**Important qualification for this project.** Holloway's restriction is *purposive*, not arbitrary: the
small sets are built so that (say) only voicing, or only place, distinguishes their members. That makes
this the most sharply designed restriction in the branch, and also the most awkward one, because the
subsets are deliberately *non-representative* of the master. Two consequences:

- It is an unusually powerful test of IIA. If odds between two consonants change when the other members
  of the set stop sharing a dimension with them, proportional renormalization fails for a structural
  reason, not a noise reason.
- It is a *hard* test to score with a zero-parameter model, because a Thurstonian account needs the
  dimensional geometry (correlated latents), not just marginal discriminabilities. Lee (1968) computed
  precisely this: constancy differs between univariate and bivariate configurations, and between
  "symmetric" and "orthogonal" bivariate layouts. Holloway's design is the empirical counterpart to
  Lee's Fig. 1 cases (4), (5) and (6). **These two papers should be read together and, as far as I can
  tell, never have been** — Lee (1968) has only two citing works and Holloway (1970) is not one of them.

## What numbers are printed (master matrix? restricted matrices? counts or proportions? which table numbers?)

NOT VERIFIED (paywalled, 8 pages). Assessment:

- The abstract says "*performance is estimated in terms of the discrimination of each relevant
  dimension*" — that is, the headline numbers are **per-dimension discrimination indices, not raw
  confusion matrices**. That is a warning sign: this may be a paper that manipulates the response set
  but publishes derived per-dimension statistics rather than the matrices.
- Against that: Holloway (1971) states verbatim that "*A new procedure for data analysis is applied to
  the results of an experiment reported by the author*", which means the raw stimulus-response data of
  the 1970 experiment were available to him in cell-level form and, quite likely, printed in 1970 —
  otherwise the 1971 paper would have had to re-print them.
- Holloway (1968), the immediate predecessor, does discuss "*The S/R matrix*" explicitly, so this
  author's habit is to print matrices.
- Eight pages of QJEP is enough for a couple of small consonant matrices but not for a large family.
- Table numbers unknown.

**Net: 50/50 whether the matrices are printed. This is the paper whose contents I am least able to
predict and it should be checked early, cheaply, before any large acquisition budget is committed.**

## Access (a DIRECT url you have fetched, plus whether it is open, paywalled or Wayback-only)

- `https://api.crossref.org/works/10.1080/14640747008401921` — FETCHED 200. **Open.** Full author
  abstract (quoted above), affiliation, volume 22, pages 467-474.
- `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=5470326&retmode=xml` —
  FETCHED 200. **Open.** Confirms title, journal, pages 467-74. **No abstract** in MEDLINE.
- `https://api.unpaywall.org/v2/10.1080/14640747008401921?email=...` — FETCHED 200. `is_oa: false`,
  `oa_status: "closed"`, zero OA locations.
- `https://www.tandfonline.com/doi/abs/10.1080/14640747008401921` — FETCHED, **HTTP 403** (Cloudflare
  "Just a moment..."). **Paywalled.**
- `https://journals.sagepub.com/doi/10.1080/14640747008401921` — FETCHED, **HTTP 403** (Cloudflare).
  **Paywalled.** (QJEP of this vintage is served from both hosts.)
- `https://pubmed.ncbi.nlm.nih.gov/5470326` — listed as a location by OpenAlex; PubMed record fetched
  via E-utilities as above, abstract-free.
- `https://api.openalex.org/works/https://doi.org/10.1080/14640747008401921` — FETCHED 200. Full
  17-item reference list retrieved. Only **2 citing works**: Neisser (1977) "The hidden preattentive
  processes", *American Psychologist*; and Holloway (1971) itself. **This paper has been read by almost
  nobody.**
- Wayback CDX for `*tandfonline.com*14640747008401921*` — FETCHED, **no snapshots**. Not even
  Wayback-only.
- MRC CBU publications site (`https://www.mrc-cbu.cam.ac.uk/publications/`) — FETCHED 200, but its
  search endpoint returned 404 and no Holloway records were surfaced. The APU/CBU does not appear to
  host a green copy.

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**NEEDS LIBRARY ACCESS. Priority: medium-high, and check it early because it is cheap and its content
is uncertain.** Eight pages behind a single Taylor & Francis / SAGE paywall. Order it together with
Holloway (1968) and Holloway (1971) — three papers, one author, two publishers, and the 1971 paper is
the one that reports the failure.

If the matrices are printed, this becomes one of the two best datasets in the branch (with Pollack &
Decker 1960), because the dimensional control makes it the only auditory experiment that can
discriminate a *correlated-latent* Thurstonian model from an independent-latent one — Lee's (1968)
"symmetric" versus "orthogonal" cases. If only per-dimension indices are printed, mark it unusable in
one line and rely on Holloway (1971) for the verdict.

## What the authors concluded about CRR, quoted verbatim where possible

Holloway does not name the constant-ratio rule in the abstract; his target is perceptual tuning. The
abstract in full, verbatim:

> "The perceptual tuning hypothesis can be expressed as a prediction that discrimination of a particular
> stimulus dimension will be enhanced if it is the only dimension requiring discrimination and not one of
> a set of dimensions in a multidimensional discrimination task. In previous tests of the tuning
> hypothesis, using speech material, the size of a message set has been manipulated as an indirect means
> of controlling the number of dimensions an observer is required to discriminate. In the present
> experiments the particular items appearing in a message set have been selected in order to control
> precisely the number of dimensions relevant to the discrimination task, and performance is estimated in
> terms of the discrimination of each relevant dimension. No evidence is found for a perceptual tuning
> effect even though it is argued that the conditions of the present experiment represent those most
> favourable for its elicitation."

Read for CRR content: **"No evidence is found for a perceptual tuning effect" is a result *consistent
with* the CRR**, in the sense that shrinking the response set did not sharpen discrimination of the
surviving dimension. Holloway is therefore, on this measure, a negative result for context-dependence
and thus a point in the CRR's favour — the opposite polarity from his 1971 paper.

That tension is the interesting thing about Holloway and worth stating carefully in the paper: on a
*marginal, per-dimension* measure (1970) he finds no set-size effect, and on a *joint, cell-level*
measure (1971) he finds a reliable dependency. Verbatim from the Holloway (1971) abstract:

> "Several investigations of the perception of consonants spoken in noise have purported to show the
> independence of the linguistic dimensions which define a consonant. A new procedure for data analysis
> is applied to the results of an experiment reported by the author. This analysis suggests that there is
> a small but reliable dependency effect. Application of the present technique to the data of Miller and
> Nicely (1955) also shows a significant dependency effect."

Marginals conform; the joint distribution does not. That is precisely the argument for scoring
distributional forecasts instead of scalar summaries, and Holloway 1970/1971 is the cleanest historical
instance of it in the auditory literature.
