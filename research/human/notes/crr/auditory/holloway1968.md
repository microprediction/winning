# Holloway (1968) — two sets of four words, bidimensional S/R matrix analysis; independence not IIA

## Citation

Holloway, C. M. (1968). Perceptual Independence in the Perception of Speech. *Quarterly Journal of
Experimental Psychology*, **20**(4), 336-350. doi:10.1080/14640746808400173
Medical Research Council Applied Psychology Unit, Cambridge (the APU — same laboratory as Morgan,
Chambers & Morton).

This paper is not in the task brief's spine, but it belongs there: it is the first of Holloway's three
APU papers on the topic (1968, 1970, 1971) and it is the one that establishes the analytic apparatus —
the "independence model" — that Holloway (1971) then turns against the CRR literature. Recovered from
the OpenAlex reference list of Holloway (1971).

## Stimuli and master response set

**Two sets of four words each**, chosen so that each set is phonetically describable in terms of two
dimensions; presented in noise. Verbatim from the abstract:

> "Two sets of four words were chosen which could be considered phonetically to be describable in terms
> of two dimensions. The S/R matrix was analysed as if the stimuli were bi-dimensional elementary
> stimuli."

So the masters are **two 4x4 stimulus-response matrices**, each with a 2x2 factorial internal
structure.

Secondary datasets reanalysed: **Miller & Nicely (1955)** (16 English consonants in noise and under
filtering; JASA 27, 338-352, doi:10.1121/1.1907526) and **Conrad (1964)** (acoustic confusions in
immediate memory). Verbatim: "*Two analyses were conducted upon data from Miller and Nicely (1955) and
Conrad (1964) to discover whether the perception of phonemes was also predictable on the independence
model.*"

## Restricted response sets (state whether nested, overlapping, or a relabelling)

**Neither nested nor a relabelling in the CRR sense — there is no master-to-subset prediction here.**
The two 4-word sets are separate, parallel designs, not a master set and a restriction of it.

What Holloway restricts instead is the *model*: he tests whether the 4x4 matrix factorises into two
independent 2-way dimensional decisions (perceptual independence / separability), which is a different
axiom from IIA. The relation to the CRR is indirect but real: independence of dimensions is the
mechanism the consonant-in-noise literature (Miller & Nicely 1955; Pollack & Decker 1960) had offered
as the *reason* the CRR should hold, so refuting independence undercuts the CRR's rationale. That is
exactly the move Holloway (1971) makes.

**Verdict for this project's inclusion criterion: this paper does NOT report a master matrix plus a
restricted response set over the same stimuli.** It reports two independent 4x4 masters. It is included
here as essential context for Holloway (1970) and (1971), not as a CRR dataset.

## What numbers are printed (master matrix? restricted matrices? counts or proportions? which table numbers?)

NOT VERIFIED (paywalled, 15 pages). The abstract's phrase "*The S/R matrix was analysed*" (singular,
definite) strongly implies at least one full stimulus-response matrix is printed, and 15 pages of QJEP
is ample room for two 4x4 matrices plus the reanalyses. Table numbers unknown.

Note that the reanalysed source matrices are themselves freely available elsewhere: **Miller & Nicely
(1955) is the most widely reproduced confusion-matrix dataset in existence** and its 16x16 matrices
have been re-typeset many times (e.g. in the Allen-group re-analyses; a copy of
`50YearsLate-RepeatingMillerNicely55.06.pdf` was already present in this session's working directory).
So the Miller & Nicely part of Holloway (1968) can be reproduced without buying the paper.

## Access (a DIRECT url you have fetched, plus whether it is open, paywalled or Wayback-only)

- `https://api.crossref.org/works/10.1080/14640746808400173` — FETCHED 200. **Open.** Full author
  abstract (quoted above), volume 20, pages 336-350.
- `https://api.unpaywall.org/v2/10.1080/14640746808400173?email=...` — FETCHED 200. `is_oa: false`,
  `oa_status: "closed"`, zero OA locations.
- Publisher: QJEP of this vintage sits on Taylor & Francis (`tandfonline.com/doi/abs/10.1080/...`) and
  is mirrored to SAGE. **Both hosts returned Cloudflare HTTP 403** to curl and to WebFetch from this
  environment (verified on the sibling DOI 10.1080/14640747008401921). **Paywalled.**
- Wayback: no snapshots found for `tandfonline.com` URLs carrying the sibling Holloway DOI; none
  checked successfully for this one.
- Supporting dataset that IS open: `https://api.crossref.org/works/10.1121/1.1907526` — FETCHED 200,
  returns the complete Miller & Nicely (1955) abstract.

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**NEEDS LIBRARY ACCESS — and low priority for this project, because it is not a restricted-response-set
experiment.** Buy it only as part of a single Holloway order alongside the 1970 QJEP paper (same
publisher, same author, adjacent volumes), where it serves to explain what Holloway's "independence"
analysis actually computes. Its own data (two 4x4 masters) cannot test IIA because there is no
restriction to compare against.

## What the authors concluded about CRR, quoted verbatim where possible

Holloway does not mention the constant-ratio rule in this abstract. His conclusion is about
independence, verbatim and in full:

> "The purpose of the experiments and analyses was to discover whether the 'independence' model, which
> successfully accounts for the perception of synthetic multi-dimensional stimuli, would also describe
> the perception of speech in noise. Two sets of four words were chosen which could be considered
> phonetically to be describable in terms of two dimensions. The S/R matrix was analysed as if the
> stimuli were bi-dimensional elementary stimuli. Satisfactory fits were obtained by the independence
> model. Two analyses were conducted upon data from Miller and Nicely (1955) and Conrad (1964) to
> discover whether the perception of phonemes was also predictable on the independence model. In general
> this was found to be the case."

So in 1968 Holloway concluded **for** independence. Three years later, with a better test, he reversed
himself on the same data — verbatim from the Holloway (1971) abstract:

> "Several investigations of the perception of consonants spoken in noise have purported to show the
> independence of the linguistic dimensions which define a consonant. A new procedure for data analysis
> is applied to the results of an experiment reported by the author. This analysis suggests that there
> is a small but reliable dependency effect. Application of the present technique to the data of Miller
> and Nicely (1955) also shows a significant dependency effect."

That 1968-to-1971 reversal by a single author on a single dataset, caused purely by adopting a test with
power, is the same story as Clarke (1957) -> Morgan (1974). Worth noting as a pattern: in this
literature, every claim of axiom-conformity that was later tested properly failed.
