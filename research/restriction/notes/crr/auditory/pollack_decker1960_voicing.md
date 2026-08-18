# Pollack & Decker (1960b) — 8 consonants, response set collapsed to voiced/unvoiced (a relabelling)

## Citation

Pollack, I., & Decker, L. (1960). Perception of Consonant Voicing in Noise. *Language and Speech*,
**3**(3), 155-163. doi:10.1177/002383096000300304
Operational Applications Laboratory, Air Force Cambridge Research Center, Bedford, Massachusetts.

Companion paper to Pollack & Decker (1960a), "Consonant Confusions and the Constant Ratio Rule",
*Language and Speech* 3(1), 1-6 — same authors, same journal volume, same year. Cited in the
reference list of Holloway (1971), which is how I found it.

Included in this directory because it is the branch's clearest instance of a **coarsening
restriction** — the third kind of response-set manipulation, distinct from nesting.

## Stimuli and master response set

Eight English consonants in noise: **/p, b, t, d, f, v, s, z/**, spoken in three positions (initial,
intervocalic, final), under two masking noises (white; low-frequency).

The *stimulus* set is 8. The *response* set is **2**: the listener reports only voicing class.
Verbatim from the abstract:

> "The listeners' task was to report whether the consonant spoken was of the class /b, d, v, z/
> (voiced) or of the class /p, t, f, s/ (unvoiced)."

Factors crossed, verbatim: "*(1) the position of the consonant in the test utterance: initial,
intervocalic or final; (2) the place of articulation: alveolar /t, d, s, z/, or labial, /p, f, b, v/;
(3) the degree of occlusion: stop, /p, b, t, d/, or fricative, /f, v, s, z/, and (4) the spectrum of
the masking noise: white noise or low-frequency noise.*"

## Restricted response sets (state whether nested, overlapping, or a relabelling)

**A relabelling — specifically a coarsening / partition of the response set.** The 8 stimuli remain,
but the 8 response alternatives are merged into 2 equivalence classes {b,d,v,z} and {p,t,f,s}.

This is *not* the nested restriction the CRR was formulated for, and that is exactly why it is
interesting: proportional renormalization (Luce) and Gaussian renormalization (Thurstone) make
different predictions when a response set is *aggregated* rather than *pruned*. Under Luce's axiom the
probability of the voiced class is the sum of the four voiced item-strengths over the total, which is
a strong prediction directly testable against the 8x8 master of the companion paper. Under a
Thurstonian max-of-latents account the class probability is the probability that the max over four
correlated Gaussians exceeds the max over the other four, which is not the same functional. **The pair
(Pollack & Decker 1960a, 1960b) is therefore a natural aggregation test using the same laboratory,
the same year, the same authors, and an overlapping consonant inventory** — although note the two
inventories are NOT the same: 1960a uses {f, h, l, r, w, y, hw, #}, 1960b uses {p, b, t, d, f, v, s,
z}. Only /f/ is common. So the aggregation test is *within* 1960b only, and requires 1960b to print an
8x8 (or at least 8x2) matrix.

The other three factors (position, place, occlusion) are stimulus-side manipulations, not response-set
restrictions.

## What numbers are printed (master matrix? restricted matrices? counts or proportions? which table numbers?)

NOT VERIFIED (paywalled, 9 pages). What the abstract implies:

- The reported dependent variable is voicing-detection accuracy broken down by position x place x
  occlusion x noise. That reads as **percent-correct tables, not confusion matrices**: with a 2-choice
  response the "matrix" is 8x2 at best, and more likely reported as a hit rate per consonant.
- There is **no evidence of an 8x8 master matrix in this paper**. The 8x8 masters in this programme
  belong to the companion paper (1960a) and to Miller & Nicely (1955).
- No mention of the constant-ratio rule in the abstract; the framing is entirely about acoustic cues
  to voicing ("*low-frequency cues to voicing which are independent of place of articulation and
  high-frequency cues which vary with place of articulation*").
- Table numbers unknown.

**One-line verdict on printed numbers, as the brief requests for this class of paper: this paper
manipulates the response set (by collapsing it to two classes) but appears to publish only
voicing-accuracy percentages rather than a master-plus-restricted matrix pair, so it is probably
unusable.**

## Access (a DIRECT url you have fetched, plus whether it is open, paywalled or Wayback-only)

- `https://api.crossref.org/works/10.1177/002383096000300304` — FETCHED 200. **Open.** Full author
  abstract (quoted above), volume 3, pages 155-163.
- `https://api.unpaywall.org/v2/10.1177/002383096000300304?email=...` — FETCHED 200. `is_oa: false`,
  `oa_status: "closed"`, zero OA locations.
- Sage landing page under `https://journals.sagepub.com/doi/10.1177/002383096000300304` — Sage returns
  Cloudflare **403** to both curl and WebFetch from this environment (verified on the sibling DOI
  `10.1177/002383096000300101`). **Paywalled.**
- Wayback: no snapshots found for any `sagepub.com` URL carrying this DOI (CDX queries on the sibling
  *Language and Speech* DOIs all returned empty).

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**PROBABLY UNUSABLE — deprioritise.** Manipulates the response set only by coarsening it to two
classes, and on the evidence of its abstract reports voicing-accuracy percentages rather than
matrices. Do not buy this paper for its own sake.

Reconsider only in one scenario: if Pollack & Decker (1960a) turns out to print full 8x8 masters and
someone wants an *aggregation* rather than *pruning* test of the axiom. In that case check whether
1960b prints per-consonant voicing hit rates in enough detail to be reconciled with an 8x8 from the
same laboratory — but the inventories barely overlap (/f/ only), so even then the payoff is thin.
Miller & Nicely (1955), which uses all 16 consonants and *is* widely available, is a far better
substrate for aggregation tests.

## What the authors concluded about CRR, quoted verbatim where possible

**The authors say nothing about the constant-ratio rule in this paper's abstract.** Their conclusion
is about acoustic cues, verbatim:

> "The absence of voicing was perceived better in alveolar consonants than in labials in low-frequency
> noise. Otherwise there were no large effects of position, place of articulation, or degree of
> occlusion, on voicing perception. The results are interpreted in terms of low-frequency cues to
> voicing which are independent of place of articulation and high-frequency cues which vary with place
> of articulation."

The paper enters the CRR story only indirectly: Holloway (1971) cites it alongside Pollack & Decker
(1960a), Clarke (1957), Clarke (1959) and Clarke & Anderson (1957) in the reference list of "A Test of
the Independence of Linguistic Dimensions", i.e. as part of the corpus of consonant-in-noise studies
that "*have purported to show the independence of the linguistic dimensions which define a
consonant*" (Holloway 1971 abstract) — the independence claim that Holloway then showed to fail.
