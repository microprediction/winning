# Auditory / speech branch of the constant-ratio-rule literature — index and triage

Compiled 2026-08-17/18. One file per experiment or document; each uses the fixed heading set
(Citation / Stimuli and master response set / Restricted response sets / What numbers are printed /
Access / Usability verdict / What the authors concluded about CRR).

The constant-ratio rule (CRR) is Luce's proportional renormalization under another name: removing
response alternatives is asserted to preserve the odds between survivors. Clarke named it in 1957;
Luce (1959) showed it to be part (i) of his choice axiom and cited Clarke's data as its only empirical
support.

## Headline findings

**1. Nothing in this branch was ever framed as a Gaussian-versus-Gumbel forecasting contest, but the
pieces were all present by 1968.**
- Clarke (1959) ran the contest on the *wrong statistic*: threshold model vs CRR vs "the theory of
  signal detectability", scored on percent correct. Verdict: threshold fails, and CRR and signal
  detectability are **both compatible** with the speech data. Percent correct has no power to separate
  them. See `clarke1959.md`.
- Lee (1968) computed exactly what a Gaussian mechanism implies about CRR constancy, with no free
  parameters, and found violations up to a factor of 4.5 (constancy 0.22 at d' = 3). He then asked the
  field to publish cell-level data instead of "gross plots and gross statistics". **He was cited twice.**
  See `lee1968.md`.
- Nobody combined the two.

**2. Every claim of axiom-conformity that was later tested with a proper significance test failed.**
- 1957-1962: Clarke, Clarke & Anderson, Egan, Anderson, Pollack & Decker, Hodge & Pollack all conclude
  FOR the rule, using eyeball criteria (Clarke's .10 absolute deviation; Pollack & Decker's 4-percentage-
  point average). Hodge (1967, p. 430) states flatly that "a satisfactory statistical test is not
  available (Clarke, 1957)".
- 1971: Holloway applies a test with power to his own data and to Miller & Nicely (1955) and finds "a
  small but reliable dependency effect" — having himself concluded FOR independence in 1968.
- 1973: Morgan, Chambers & Morton reject the cross-ratio (equal-confusability) null on four 9x9 digit
  matrices with chi-squares of 1121.62, 2619.98, 375.26, 274.76 on 55 df.
- 1974: Morgan applies a likelihood-ratio test to Clarke (1957) and Egan (1957) — the founding data —
  and finds both **depart significantly** from the CRR.
- 1982: Townsend & Landon find the CRR inferior to a free-parameter similarity choice model on visual
  letters.

**3. The failures are signed and patterned, not random.** CRR under-predicts accuracy on small, easy
restricted sets. Hodge (1967) Table 3: positive algebraic diagonal differences in 6 of 8 auditory 2x2
conditions, and 28.6% / 21.4% / 50.0% of cells exceeding Clarke's own .10 criterion in three of them.
Morgan (1973, p. 382) predicts the same direction from first principles. Lee (1968) predicts d'-dependent
signed violations from Gaussian machinery alone. This is the shape of a beatable forecast, not a coin flip.

**4. Almost none of the matrices are open.** Of the fourteen documents catalogued, exactly **one** is
open, complete and legible: Morgan, Chambers & Morton (1973), four 9x9 digit matrices — transcribed and
arithmetic-checked in `morgan1973.md`. And it has **no restricted response sets**.

**5. The one paper that both prints the matrices and has restricted sets, in every case, is paywalled.**

## Triage table

| File | Document | Master + restricted? | Matrices printed? | Access | Verdict |
|---|---|---|---|---|---|
| `morgan1973.md` | Morgan, Chambers & Morton 1973, *P&P* 14, 375-383 | **No** (four 9x9 masters only) | **Yes, counts, Tables 1-4** | **OPEN (bronze)**, in hand | **USABLE NOW** — transcribed here |
| `hodge1967.md` | Hodge 1967, *P&P* 2, 429-437 | Yes (8->4, 8->2; visual/kinesthetic) | **No** — summary stats + unlabelled scatterplots | **OPEN (bronze)**, in hand | Unusable as matrices; usable as the free carrier of Hodge & Pollack's auditory summary stats |
| `lee1968.md` | Lee 1968, *P&P* 4, 217-219 | Hypothetical 3->2 | No data at all; Table 1 = model calculations | **OPEN (bronze)**, in hand | Usable as theory; unusable as data |
| `morgan1973a_clusters.md` | Morgan 1973a, *P&P* 13, 13-24 | No (26x26 + 35x35 masters) | **Yes** — but scan illegible | **OPEN (bronze)**, in hand | **NEEDS BETTER SCAN** |
| `clarke1957_exp1.md` | Clarke 1957 Exp I — CVs, three 6x6 + six 3x3 | **Yes, nested** | Almost certainly (Morgan re-tested it) | Paywalled ($30 AIP) | **NEEDS LIBRARY ACCESS** |
| `clarke1957_exp2.md` | Clarke 1957 Exp II — monosyllables | **Yes, nested** | Almost certainly | Same article | Needs library access |
| `clarke1957_exp3.md` | Clarke 1957 Exp III — digits | **Yes, nested** | Almost certainly | Same article | Needs library access |
| `clarke_anderson1957.md` | Clarke & Anderson 1957, *JASA* 29, 1318-1320 | **Yes, 10 -> 5+5, naive Ss** | Cell-level proportions; 10x10 master uncertain | Paywalled, no Wayback | Needs library access |
| `egan1957.md` | Egan 1957, *JASA* 29, 482-489 | **Yes, message-set sweep** | **Yes** ("confusion matrices ... were also determined") | Paywalled | **NEEDS LIBRARY ACCESS — joint top priority** |
| `pollack_decker1960.md` | Pollack & Decker 1960a, *L&S* 3, 1-6 | **Yes, 8x8 -> three overlapping 4x4, swept over S/N** | Yes, proportions | Paywalled, no Wayback | **NEEDS LIBRARY ACCESS — top three** |
| `hodge1962.md` | Hodge & Pollack 1962, *JEP* 63, 129-142 | **Yes, 8->4 and 8->2, nine conditions** | Likely; summary stats recovered free | Paywalled (APA) | Summary stats **USABLE NOW** (transcribed); matrices need library access |
| `morgan1974.md` | Morgan 1974, *JMP* 11, 107-123 | Reanalysis of the whole corpus | Unknown — **may reproduce Clarke's and Egan's matrices** | Paywalled (Elsevier); try Kent repository | **NEEDS LIBRARY ACCESS — acquire FIRST** |
| `holloway1970.md` | Holloway 1970, *QJEP* 22, 467-474 | **Yes, two levels of decision complexity** | 50/50 — abstract promises per-dimension indices | Paywalled, no Wayback | Needs library access; check early, cheap |
| `holloway1971.md` | Holloway 1971, *L&S* 14, 326-340 | Reanalysis (Holloway 1970 + Miller & Nicely 1955) | Test statistics yes; matrices unknown | Paywalled | Miller & Nicely half usable now from open data |
| `holloway1968.md` | Holloway 1968, *QJEP* 20, 336-350 | **No** (two parallel 4x4 masters) | Likely one S/R matrix | Paywalled | Low priority — not a restriction study |
| `clarke1959.md` | Clarke 1959, *JASA* 31, 835 | Yes (n swept) but | **No — percent correct only** | Bronze OA at AIP (403 to bots); full text = the abstract, obtained | **UNUSABLE as data**; conceptually the most important abstract in the branch |
| `pollack_decker1960_voicing.md` | Pollack & Decker 1960b, *L&S* 3, 155-163 | **Relabelling** (8 stimuli -> 2 voicing classes) | Probably voicing accuracy only | Paywalled | Probably unusable — deprioritise |
| `green_birdsall_macnee1958.md` | Green, Birdsall & Macnee 1958, Michigan | Yes (vocabulary size) | Probably articulation score only | **GREEN OA, URL verified, Cloudflare-blocked to bots** | **Open in a browser — 2 minutes, real upside** |
| `egan1957_techreport5750.md` | Egan 1957, Tech Report 57-50, Indiana | Presumed yes | Title promises matrices | **Not locatable** | Chase only after the journal version |
| `anderson1959.md` | Anderson 1959, AFCRC-TN-58-60, Indiana | Presumed yes (visual monosyllables) | Unknown | **Not locatable** | Low priority — wrong modality |
| `carterette_wyman1961.md` | Carterette & Wyman 1961, Psychonomic Society talk | Presumed yes | **Nothing published** | Nonexistent | **UNUSABLE — dead end** |

## Recommended acquisition order

1. **Morgan (1974), *JMP* 11, 107-123.** May reproduce Clarke's and Egan's matrices *with* the
   significance tests already computed — two datasets and the key result for one purchase. Also cites
   Thurstone (1927) alongside Luce, so it is the natural paper to position this work against. Try the Kent
   Academic Repository first (Morgan was at Kent).
2. **Clarke (1957), *JASA* 29, 715-720.** The origin. Three experiments, nested restrictions, and the only
   evidence Luce ever cited for his axiom.
3. **Egan (1957), *JASA* 29, 482-489.** The other dataset Morgan broke. Eight pages, so good odds of full
   matrices. Effectively invisible in CRR reviews because its title omits the rule.
4. **Pollack & Decker (1960a), *L&S* 3, 1-6.** Master set fully specified in the open abstract
   ({f, h, l, r, w, y, hw, #}); three **overlapping** 4x4 subsets giving redundant constraints on the same
   odds; and an **S/N sweep** giving a discriminability axis. The closest thing to a designed
   Gaussian-vs-Gumbel experiment in the 1960s.
5. **Hodge & Pollack (1962), *JEP* 63, 129-142.** Potentially the richest: nine 8x8 masters with 4x4 and
   2x2 subsets and spacing/range under experimental control. Tones, not speech.
6. **Holloway (1970) + (1971) + (1968)**, one order. The dimensional control in 1970 is the only auditory
   design that could separate correlated- from independent-latent Thurstonian models — Lee's (1968)
   "symmetric" vs "orthogonal" cases.
7. **Better scan of Morgan (1973a)**, or go direct to Hull (1973) *BJP* 64, 579-585 and Conrad (1964) *BJP*
   55, 75-84 for the 35x35 and 26x26 auditory masters. Note Hull's STM half used "two selected vocabularies
   of nine letters and the digits 1-9" — possibly a genuine restriction study.
8. **Green, Birdsall & Macnee (1958)** — free, two minutes, do it opportunistically.

## Free-now action items requiring no acquisition

- Fit zero-parameter Gaussian and Gumbel models to the four 9x9 digit matrices in `morgan1973.md` and
  compare out-of-sample scores. No paywall, no digitizing.
- Recompute Holloway's (1971) independence test on Miller & Nicely (1955), whose matrices are open and
  widely re-typeset.
- Quote the Hodge & Pollack (1962) CRR failure profile from the table in `hodge1962.md` (transcribed from
  the free Hodge 1967 PDF).
- Quote Lee (1968) Table 1 as the zero-parameter Gaussian prediction of IIA violation
  (`lee1968.md`).

## Method notes and environment limitations

Everything reported was fetched. Working APIs from this environment: Crossref
(`api.crossref.org`), Unpaywall, OpenAlex (rate-limits aggressively at ~1 req/25 s), Semantic Scholar
Graph, PubMed E-utilities, Wayback CDX, and Wayback content retrieval (intermittent HTTP 503 — retry
loops of 6-25 attempts were needed and did eventually succeed).

Blocked by bot protection (HTTP 403, Cloudflare) to both curl and WebFetch: **pubs.aip.org**,
**journals.sagepub.com**, **tandfonline.com**, **psycnet.apa.org**, **link.springer.com** (after repeated
requests; Wayback copies worked instead), **deepblue.lib.umich.edu** (including its OAI-PMH and REST
endpoints), **hathitrust.org**, **researchgate.net**, **colab.ws**, **api.fatcat.wiki** (timeouts).

**General web search was unavailable.** The session's WebSearch quota was exhausted before this task
began, and every scraping alternative failed: DuckDuckGo HTML served a CAPTCHA, DuckDuckGo lite returned
HTTP 202 with an empty body, searx.be served browser verification, Mojeek returned only its own
navigation, and a scripted Bing client silently dropped quoted phrases and returned dictionary
definitions. Consequently **the grey literature (Egan TR 57-50, Anderson AFCRC-TN-58-60, Carterette &
Wyman 1961) could not be swept**, and DTIC's public search returned `"results":[]` for every query tried.
Those three items should be re-attempted with a normal browser before being written off.

The literature map itself was built without search engines, from: the reference lists of the three
obtained PDFs (Hodge 1967, Lee 1968, Morgan/Chambers/Morton 1973) plus Townsend & Landon (1982, also in
hand), and from OpenAlex forward-citation and backward-reference queries. That route is what surfaced the
spine members missing from the original brief: **Egan (1957) and its technical report, Clarke (1959),
Anderson (1959), Carterette & Wyman (1961), Holloway (1968), Pollack & Decker (1960b), Hull (1973),
Conrad (1964), Green/Birdsall/Macnee (1958), and Pollack's message-uncertainty series.**

Two citation corrections to the original brief: **Lee (1968) is *Perception & Psychophysics* 4(4),
217-219, not *Psychonomic Science***; and **Holloway (1971) is *Language and Speech* 14(4), 326-340.**
