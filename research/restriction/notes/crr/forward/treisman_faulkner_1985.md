# Treisman & Faulkner 1985 — "On the choice between choice theory and signal detection theory"

Status: **resolved as far as it can be without a library copy.** The threat is real but much
smaller than `PRIOR_ART.md` currently assumes. It is category (B), and on the strongest
available secondary description it is a *weaker* (B) than Robinson et al. — T&F found
**neither** model's parameter invariant and picked the winner by a plausibility argument
about the sign of the drift, not by predictive accuracy.

## Citation

Treisman, M., & Faulkner, A. (1985). "On the Choice between Choice Theory and Signal
Detection Theory." *The Quarterly Journal of Experimental Psychology Section A* 37(3):
387–405. Published August 1985. doi:[10.1080/14640748508400941](https://doi.org/10.1080/14640748508400941).

Both authors: Department of Experimental Psychology, University of Oxford. (Note the DOI
prefix is Taylor & Francis' `10.1080`, but the DOI now resolves to SAGE — QJEP moved
publishers and SAGE holds the back file. Crossref lists the publisher as SAGE Publications.)

Citation counts: **7** (Crossref), **12** (OpenAlex), **15** (Semantic Scholar). This is a
very lightly cited paper.

Full abstract, verbatim, as carried in the publisher's JATS metadata and retrieved through the
CORE API:

> "Signal detection theory and choice theory are sufficiently similar in their predictions to
> make it difficult to decide which gives the better fit to experimental data. In some cases,
> such as the analysis of word recognition, this similarity has allowed choice theory
> formulations to be employed as approximations to those given by signal detection theory. It
> could be proposed that signal detection theory gives the more valid description of the
> underlying processes, and choice theory provides an approximation to it; or one might choose
> to argue the reverse. A procedure for testing between choice theory and signal detection
> theory is described and is applied to the data of Miller, Heise and Lichten (1951) and to
> the results of an experiment. Both sets of results favour signal detection theory rather
> than choice theory."

## Access attempted and outcome

**Full text NOT obtained.** Every route below was tried in this pass. Routes are listed so the
next pass does not repeat them.

| Route | Outcome |
|---|---|
| Crossref API (metadata + full-text links) | DOI, pagination, affiliations, and the two `text-mining`/`similarity-checking` link targets recovered. Both point at `journals.sagepub.com/doi/pdf/...`. |
| `doi.org` resolution | Resolves to SAGE. |
| `journals.sagepub.com/doi/10.1080/...` (landing) | **HTTP 403**, Cloudflare managed challenge (`cf-mitigated: challenge`). |
| `journals.sagepub.com/doi/pdf/10.1080/...` | **HTTP 403**, Cloudflare challenge. |
| `www.tandfonline.com/doi/abs/10.1080/...` (pre-move publisher) | **HTTP 403**, Cloudflare challenge. |
| Text-extraction proxy (`r.jina.ai`) against the SAGE landing page | Returned the Cloudflare "Just a moment..." interstitial; target returned 403. |
| Unpaywall | `is_oa: false`, `oa_status: closed`, `has_repository_copy: false`, `oa_locations: []`. Definitive: no OA copy is registered anywhere. |
| OpenAlex | Lists a second location — **UCL Discovery, `discovery.ucl.ac.uk/20034/`** (Faulkner later moved to UCL). Pursued; see next row. |
| UCL Discovery record (`/id/eprint/20034/`) | **Cloudflare challenge**, both via curl and via WebFetch (403). Unpaywall's `has_repository_copy: false` indicates this record is metadata-only in any case, so the block probably costs nothing. |
| CORE API v3 | Record found (core_id 280927402). Carries the **full abstract** (quoted above). `fullText: "Not available for public API users."`, `downloadUrl: ""`. |
| Semantic Scholar Graph API | Record found. `openAccessPdf` status `BRONZE` but the URL is the same blocked SAGE PDF; `abstract: null` (elided by publisher). |
| Semantic Scholar **citation contexts** | Productive — see "Citing literature" below. |
| Wayback CDX, `journals.sagepub.com/doi/pdf/...` | Two captures (2019), both **redirect stubs only** (301 / 302, `3I42H3S6NNFQ2MSVX7XZKYAYSCX5QBYJ` = empty body). No content. |
| Wayback CDX, `journals.sagepub.com/doi/10.1080/...` | Three captures (2018, 2024), all 301/302 redirects. Raw `id_` fetch of the 2018 capture returned 0 bytes. |
| Wayback CDX, `/doi/abs/`, `/doi/epdf/`, `tandfonline.com/doi/*`, ResearchGate slug | **No captures at all.** |
| Europe PMC | `hitCount: 0` for the DOI. QJEP Section A 1985 is not indexed. |
| PubMed | 0 hits. Not indexed. |
| archive.org full-text search / advancedsearch (QJEP volumes) | No copy of the journal; scholar.archive.org returns a session-verification wall. |
| fatcat / scholar.archive.org API | Non-JSON response; no file record. |
| ORA (Oxford, Treisman's own institution) | Search page reachable but returns no matching record. Treisman's only ORA-deposited item is his 1962 thesis, "A study of the sensory threshold". |
| Google Books API (textbook discussions) | **Quota exhausted** for this project (HTTP 429). Not tried successfully — a live lead for the next pass. |
| WebSearch | **Budget exhausted for the session** (200/200) before the first query. Not used at all in this pass. |

Two routes remain genuinely untried and are cheap: **(a)** a library / interlibrary copy —
still the recommendation; **(b)** Google Books full-text search for textbook paraphrases,
blocked only by a daily API quota. WebSearch with a fresh budget may also surface a course-page
scan.

## Data and design

**Both.** The abstract settles this: the procedure "is applied to the data of Miller, Heise and
Lichten (1951) **and** to the results of an experiment."

**The reanalysis half.** Miller, G. A., Heise, G. A., & Lichten, W. (1951), "The intelligibility
of speech as a function of the context of the test materials", *J. Exp. Psychol.* 41:329–335,
doi:10.1037/h0062491. This is the classic speech-in-noise study in which the listener's **test
vocabulary size** is manipulated (monosyllabic word vocabularies across roughly 2 to 256 items,
plus digits) at several speech-to-noise ratios. Not open access; not obtained in this pass
either, so the nesting structure of its vocabularies is **not verified here** — see the caution
in the next section.

**The experiment half.** Not described in the abstract. The best available description is
Robinson, DeStefano, Vul & Brady (2023), *J. Math. Psychol.* 137:102805 (author manuscript
obtained in full from eScholarship, `qt3fj3z8r4`), who read the paper and characterise it as:

> "We note that a similar test was used in an **auditory memory task** in an early study by
> Treisman and Faulkner (1985)."

and, on sample size:

> "Furthermore, this study only used data from **6 participants** and may have been
> underpowered."

and on the presentation format — this is the important detail:

> "The first reason we used a visual memory task is because this allows us to present all m-afc
> alternatives visually, **instead of having participants maintain these in working memory**.
> Accordingly, this study design minimizes differences in memory load across m-afc task,
> addressing the core limitation of the Treisman and Faulkner experiment [...]"

So: an auditory *m*-AFC memory task, *m* varied, alternatives held in working memory rather
than displayed, n = 6.

**Note a discrepancy with the current `PRIOR_ART.md` entry.** That entry says Robinson et al.
characterise T&F as testing invariance "on Miller, Heise & Lichten's vocabulary-size data."
The **published** JMP text does not mention Miller, Heise & Lichten at all — it describes only
T&F's own auditory memory experiment. The MHL link comes from T&F's own abstract, not from
Robinson. Worth correcting so the two claims are not run together.

## Whether the response set is genuinely restricted

**On the evidence available: no — this is set-size variation, not nested restriction over
shared alternatives, and it is not scored cell by cell.**

Three separate points, in decreasing confidence:

1. **The test is on a scalar summary, not on a matrix.** Whatever the menus were, the quantity
   compared across conditions is a single fitted number per condition — *d′* for SDT, *β* for
   the choice/softmax model. Nothing in the described procedure requires, or uses, odds between
   named surviving alternatives. This alone means the paper cannot be a CRR test in the
   project's sense: proportional renormalization is never the competing forecast, and never
   appears.

2. **The experiment half varies *m* without a master menu.** Robinson et al. treat the design
   as "variations in the number of alternatives presented at test in an m-afc task", and their
   entire criticism is that in T&F's version *m* is **confounded with memory load** because the
   alternatives were held in working memory. A design in which increasing *m* changes the task's
   memory demand is not a design in which "the surviving alternatives are unchanged" — it is
   the project's own **quality-changing-removal** failure mode, from the other direction.

3. **The MHL half is unverified and probably not nested.** MHL varied vocabulary *size*; whether
   the smaller vocabularies were subsets of the larger ones, and whether cell-level confusion
   matrices were published at all rather than percent-correct curves, is **not established
   here.** MHL is famous for intelligibility *curves* (percent correct against S/N by vocabulary
   size), which is precisely the "varies set size but publishes only percent correct" case the
   project's README tells us to note in one line and move on. Treat as **not usable, pending
   confirmation** — and note that if MHL published only percent correct, then T&F's reanalysis
   of it *could not* have been a share-level restriction test even in principle.

## Fitted or out-of-sample

**Fitted, per condition. Nothing held out. No forecast scored.**

This is the decisive finding and it comes from the only citing paper that actually engages with
the design. Robinson et al. (2023), §1.5, "Critical test: Parameter invariance across changes of
m in m-afc tasks", first describe their own method:

> "We compared the Gaussian signal detection and softmax model by examining which model's
> parameters (d′ in SDT; β in LCA/softmax) are invariant across variations in the number of
> alternatives presented at test in an m-afc task."

and then say T&F did "a similar test", reporting T&F's result as:

> "These authors reported evidence for the Gaussian signal detection model, however, **their
> results were somewhat ambiguous. Mainly, they found that variations in m-afc produces
> decreases in d′ and increases β**, parameters in the Gaussian signal detection and softmax
> model, respectively. The researchers interpreted this as evidence for the Gaussian signal
> detection model because they reasoned that increasing the number of alternatives in the
> auditory task may increase memory load and hurt performance, but not improve it. However,
> while the finding that d′ decreases with m may be more psychologically plausible, **it does
> not demonstrate that parameters of this model are invariant with m** because m is confounded
> with memory load."

Read that carefully. It says three things that matter:

- **A parameter was estimated separately within each set-size condition** — otherwise there
  would be no "decreases in d′" and no "increases β" to observe.
- **Neither parameter was invariant.** T&F's own preferred model *failed* the invariance test
  too; *d′* moved with *m*. This is not "Gaussian survives the menu change and Luce does not."
- **The winner was chosen by a psychological plausibility argument about the *direction* of
  the drift** — a decrease in sensitivity with load is believable, an increase in *β* is not.
  Not by out-of-sample accuracy, not by held-out likelihood, not by any scored prediction.

## Free parameters

**Yes — for both models, refitted within every response-set-size condition.** *d′* for the
Gaussian model, *β* for the choice/softmax model, one of each per menu.

This is the question the parent flagged as decisive, and it comes out against the prior-art
claim. There is **no parameter-free restriction map** anywhere in T&F 1985. Full-menu
quantities are never used to generate restricted-menu predictions; the restricted-menu data are
used to *estimate* the restricted-menu parameter, and the parameter estimates are then compared
across conditions by eye and by plausibility.

Caveat, stated so a referee cannot use it against us: **parameter invariance is the logical dual
of the out-of-sample forecast.** If *d′* had been invariant across *m*, then full-menu *d′*
would predict restricted-menu shares with nothing fitted. So the *idea* — "see whether the
Gaussian or the Luce parameter survives a change of response-set size, and prefer the one that
does" — is genuinely 1985 (and arguably older; see Treisman 1977 "On the stability of d′"). What
is not 1985 is the execution: a held-out forecast, scored against proportional renormalization,
on shared alternatives, at the level of cell shares.

## Conclusion quoted

The paper's own conclusion cannot be quoted from the paper. The abstract's closing sentence,
verbatim, is:

> "A procedure for testing between choice theory and signal detection theory is described and is
> applied to the data of Miller, Heise and Lichten (1951) and to the results of an experiment.
> **Both sets of results favour signal detection theory rather than choice theory.**"

The strongest secondary statement of what that amounted to, verbatim from Robinson et al.
(2023):

> "These authors reported evidence for the Gaussian signal detection model, however, their
> results were somewhat ambiguous."

## Verdict as prior art

**Category (B). Not prior art. It does not narrow the project's claim to the protocol alone,
and the current `PRIOR_ART.md` framing of it as "the highest residual risk in this file" should
be downgraded.**

Against all four of the project's criteria:

| Criterion | T&F 1985 |
|---|---|
| 1. Gaussian / Thurstonian / probit choice map | **Yes** (equal-variance Gaussian SDT). |
| 2. Calibrated on full-menu shares only | **No.** Calibrated separately within each menu. |
| 3. Restricted-menu shares predicted out of sample, nothing fitted to the target | **No.** A free parameter is fitted to every menu, including the small ones. |
| 4. Scored against proportional renormalization as the competing forecast | **No.** The competitor is Luce's choice model *as a fitted parameterisation*, not the CRR as a forecast. Renormalization is never computed. |

And the substantive result is weaker than the second-hand characterisation implied. The
second-hand claim under test was "signal detection theory beats Luce on data where
response-set size varies." What T&F appear to have found is that **both** models' parameters
drifted with set size, and SDT was preferred because *its* drift had the psychologically
plausible sign. On the project's own standard that is not a win for either map; it is a
demonstration that neither is invariant in that task, with a judgement call on top.

**Two things that cut the other way and should be conceded in print rather than left to a
referee:**

1. The *framing* is older than the project's framing. "Choice theory versus signal detection
   theory when the response set changes" is a 1985 question with a 1985 answer in SDT's favour,
   and there is a longer Treisman thread behind it (below). The paper's contribution should be
   stated as the **protocol and the scoring** — parameter-free, out of sample, against
   renormalization, on shared alternatives — and not as the discovery that Gaussian beats Luce
   under menu change. That is a narrowing, but a modest one, and it is a narrowing the project
   has already accepted for Lee (1968) and Wills et al. (2000).

2. **Robinson et al. (2023) is the closer prior art, not T&F 1985.** `PRIOR_ART.md` lists it
   as the 2022 OSF preprint; the published version is Robinson, M. M., DeStefano, I., Vul, E.,
   & Brady, T. F. (2023), "How do people build up visual memory representations from sensory
   evidence? Revisiting two classic models of choice", *J. Math. Psychol.* 137:102805,
   doi:10.1016/j.jmp.2023.102805 — free author manuscript at
   `https://escholarship.org/content/qt3fj3z8r4/qt3fj3z8r4.pdf`. Their **primary** analysis
   fixes *d′* and *β* **across** all *m*-AFC conditions and compares log-likelihood, which is a
   cross-condition constraint much closer to the project's map than anything in T&F: "We tested
   this by comparing the relative fits of the Gaussian signal detection and softmax model when
   parameters d′ and β, respectively, were fixed across all m-afc conditions." They report
   Gaussian wins (n = 30, *t*(29) = 4.26, *p* < .001, *d_z* = 0.77), plus a flexibility control
   showing the two models fit equally when parameters vary freely, plus a lower cross-condition
   SD for *d′* (.21) than for *β* (.55). Still category (B) — the parameters are fitted jointly
   to all conditions, nothing is held out, and the CRR is never the competitor — but this is
   the live modern thread and it must be cited and distinguished, prominently. Also note their
   Gumbel framing: via Holman & Marley (1974) and Yellott (1977), Luce-vs-Gaussian *is*
   Gumbel-vs-Gaussian SDT, which is the same reduction Duffy & Smith (2025) turn against us.

## Treisman's other work on this, and how the citing literature reads T&F 1985

**Does Treisman do the same thing elsewhere?** He has a long SDT-versus-choice-theory thread in
word recognition, but no second set-size invariance paper was found. Of 121 works in OpenAlex,
the relevant ones are:

- **Treisman (1971)**, "On the word frequency effect: Comments on the papers by J. Catlin and
  L. H. Nakatani", *Psych. Review* — the opening move: Catlin and Nakatani had Luce-choice-rule
  accounts of word recognition; Treisman argues for SDT. This is the debate the 1985 abstract's
  second sentence alludes to ("such as the analysis of word recognition").
- **Treisman (1977)**, "On the stability of d′", *Psych. Bulletin* — parameter-invariance
  reasoning about *d′* in its own right, eight years before T&F 1985. **The nearest ancestor of
  the invariance idea, and unread.** Worth a look before claiming the invariance framing is new.
- **Treisman (1978)**, "A theory of the identification of complex stimuli with an application to
  word recognition", *Psych. Review* 85(6):525–570 (doi 10.1037/0033-295X.85.6.525) — the SDT
  identification model that T&F 1985 is presumably testing. Closed access; abstract elided by
  APA; not obtained. **Second unread lead.**
- **Treisman (1978)**, "Space or lexicon? The word frequency effect and the error response
  frequency effect", *JVLVB*; **Treisman (2022)**, "The Word Frequency Effect: A New Theory" —
  same thread, later.
- **Treisman & Williams (1984)**, "A theory of criterion setting with an application to
  sequential dependencies", *Psych. Review* 91(1):68–111 — the criterion-setting work he is
  actually known for. It is about criterion *drift over trials*, not about menus, and is not a
  threat.

**Has anyone, 1985 to now, cited T&F 1985 as settling Gaussian-versus-Luce under set-size
change? No — exactly one paper even notices the set-size test, and it calls the result
ambiguous.** Citation contexts were pulled for all 15 Semantic Scholar citations. Every one
that mentions T&F substantively cites it for the *approximation* result (logistic ≈ Gaussian),
not for a set-size verdict:

- **Jäkel & Wichmann (2006)**, *J. Vision* 6(11):13 — the most-read citation, and it reads
  **against** the strong prior-art claim: "The few studies that compared the signal detection
  model to Luce's choice model have found that the signal detection model fits the data
  slightly better but that **Luce's choice model is in any case a very good approximation**
  (Luce, 1963, 1977; Treisman & Faulkner, 1985)."
- **Koß et al. (2024)**, *Comput. Brain & Behav.* — "there is no big difference between the
  equal-variance Gaussian model and the logistic model, and, in fact, **both can be hard to
  distinguish empirically** (Treisman & Faulkner, 1985)."
- **Iverson (2002)** — "It has long been known that Signal Detection Theory and Choice Theory
  provide **similar** estimates of sensitivity for simple forced-choice experimental designs
  (e.g., Luce, 1963; Macmillan & Creelman, 1991; Treisman & Faulkner, 1985)."
- **Carandini (2024)**, *Neuron* — cited only for "one could assume that the noise in SDT comes
  from a logistic rather than a Gaussian distribution".
- **Nakajima et al. (2019)**, *Neuron* (×2) — cited only as the source of a logistic pdf used
  for curve fitting.
- **Robinson et al. (2023)** — cites it *both* ways: for the approximation result ("the
  logistic distribution approximates the normal distribution"), and as the one prior
  set-size/invariance test, which they judge ambiguous, confounded and underpowered, and then
  redo.
- Remaining citations (Estes 1986; Christensen-Szalanski 1986; Knibb 1992; Sridharan et al.
  2014; Baker et al. 2022; Dienes 2019; Treisman 1999) carry no substantive characterisation in
  their available citation contexts. Sridharan et al. 2014 and Estes 1986 are the two whose
  full text could not be retrieved (PMC text-mining endpoints returned empty; APA paywalled);
  neither is likely to change the picture, but neither was read.

Forty years and fifteen citations, of which one engages with the design and rejects it as
inconclusive, is not a settled verdict in the field. **If a referee raises T&F 1985, the answer
is Robinson et al.'s own answer: nothing was held out, both parameters drifted, and the winner
was chosen on plausibility.**

## What would change this verdict

A library copy. Specifically, three things in it would matter:

1. **Whether T&F's "procedure for testing" is in fact a forecast rather than an invariance
   check.** The abstract calls it "a procedure for testing between" the theories, which is
   ambiguous, and Robinson et al. only report the *outcome* (parameters drifted), not the
   inferential machinery. If T&F derived a *closed-form parameter-free* prediction — e.g. the
   Gaussian *m*-AFC relation between 2-AFC and *m*-AFC performance, which is a standard
   parameter-free link — and tested it against MHL's curves, the paper moves toward category (A)
   on criteria 1 and 3 and this note is wrong. This is the single most important thing to check,
   and it is the reading the abstract's phrase "a procedure for testing between" most invites.
2. **Whether the MHL reanalysis used cell-level shares** or only percent correct. If shares,
   the MHL matrices may be a usable dataset for the project in their own right.
3. **Whether T&F's own auditory menus were nested over shared items.** If a master menu and
   nested submenus over the same items exist with printed matrices, that is a usable dataset
   regardless of what T&F concluded.

Priority: **still worth settling, but demote from "the only item that could overturn the
verdict" to third, behind Takane & Shibayama (1992) and Robinson et al. (2023).** Robinson et al.
is readable today, is closer, and is the citation a referee is more likely to have in mind.
