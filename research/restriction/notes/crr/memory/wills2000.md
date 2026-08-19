# Wills, Reimers, Stewart, Suret & McLaren (2000)

## Citation

Wills, A.J., Reimers, S., Stewart, N., Suret, M., & McLaren, I.P.L. (2000). Tests of the
ratio rule in categorization. *The Quarterly Journal of Experimental Psychology A: Human
Experimental Psychology*, 53(4), 983–1011. DOI 10.1080/713755935 (a duplicate DOI
10.1080/02724980050156263 also resolves). PMID 11131824. Received 31 July 1998;
accepted revision 11 November 1999. All authors at Cambridge University at the time.

Note: Andy Wills' own publication list drops Suret from the author string; the printed
title page and PubMed both include him. Five authors is correct.

## Domain and stimuli

Human category learning with artificial, prototype-structured visual stimuli. Each stimulus
is 12 different small symbols placed randomly on an invisible 4x3 grid inside a 3.5-cm
rectangle outline. Symbols drawn from a pool of 36 (Experiment 1) or 40 (Experiment 2)
elements, randomly allocated per subject to three categories of 12 elements each.

Training: 90 labelled exemplars, observational (no response required), each shown 2 s.
Test: unlabelled transfer stimuli in which the number of elements from the two variable
categories trades off (x and 8-x, for x = 0..8) while a third category contributes a
constant four elements. Dummy stimuli were included to obscure the design.

This is a **categorization** domain, not speech and not letters. Stimuli are meaningless
symbol arrays; category labels are A/B/C.

## Master response set and restricted response sets (nested, overlapping, or a relabelling)

This is the core of the paper and it is a **nested response-set manipulation over
identical stimuli**, run between subjects.

- Master (three-choice) set: {A, B, C}. Question at test: "Is this an A, a B, or a C?"
- Restricted (two-choice) set: {B, C} — response A disallowed. Question: "Is this a B or a C?"
- Experiment 2 adds a third condition, **novel-elements**: response set is the full
  {A, B, C}, but the four Category-a elements in every test stimulus are replaced by four
  elements never seen in training. This drives the magnitude term for the removed category
  toward a constant/zero *without* removing the response option — a soft restriction that
  the ratio rule must handle by the same renormalization arithmetic.

Experiment 1: 24 subjects, 12 per condition (three-choice vs two-choice). Category A was
always the fixed/disallowed category — not counterbalanced.

Experiment 2: 36 subjects, 12 per condition (three-choice, two-choice, novel-elements).
Label allocation counterbalanced: within each condition, three sub-groups of 4 assign
(a,b,c) = (A,B,C), (B,C,A), (C,A,B). This removes the label-bias confound of Experiment 1.

The CRR/ratio-rule test statistic is
q = [P(b : a,b,c restricted to b,c) - P(b : a,b,c)] / P(b : a,b,c),
i.e. the proportional gain in a survivor's probability when one alternative is removed.
Under the ratio rule q = n_A / (n_B + n_C), and q', P(a:a,b,c), P(a:a,b,c)' are all driven
only by (n_B + n_C). Therefore **all four empirical functions must move in the same
direction over every interval** of the category-appropriate-element index. That is the
prediction under test, and it survives adding a background-noise constant X to the
denominator (their Equations 14–16).

## What numbers are printed (which tables, counts or proportions, per subject or pooled)

No confusion matrices are printed. What is printed:

- Table 1: label-to-category allocation for the three counterbalancing sub-groups of
  Experiment 2 (design table, no data).
- Figure 3 (Experiment 1), three panels: (a) mean response probability vs number of
  category-appropriate elements, connected plot symbols; (b) q and P(A:A,B,C) vs number of
  category-appropriate elements; (c) same as (b) but folded onto "distance" from the
  4-element midpoint. Best-fitting quadratics overlaid.
- Figure 4 (Experiment 2), three panels: (a) P(a:a,b,c) for three-choice and
  novel-elements conditions vs number of Category-b elements; (b) mean response
  probability vs number of category-appropriate elements for all three conditions;
  (c) q and q' vs number of category-appropriate elements.
- All fitted quadratics are printed in the text with coefficients, F, t and p, e.g.
  Experiment 1 P(A) = -0.003c^2 + 0.021c + 0.29; Experiment 1 folded
  q = 0.04d^2 - 0.11d + 0.37 (F(2,2) = 116, p < .01); Experiment 2
  P(a:a,b,c) = -0.006b^2 + 0.037b + 0.291 (F(2,6) = 5.6, p < .05);
  P(a:a,b,c)' = -0.003b^2 + 0.016b + 0.112 (F(2,6) = 3.2, ns);
  q = 0.049c^2 - 0.674c + 2.48 (F(2,6) = 803, p < .0005);
  q' = -0.021c^2 + 0.244c - 0.368 (F(2,6) = 17, p < .005).
- Figures are **pooled means across subjects** (9 data points per function). The authors
  explicitly defend using means and explain why a within-subject q is undefined when a
  subject makes only Category-a responses at some level.

Crucially, the **raw trial-level data for Experiment 2 are published** (see Access), so
per-subject 3x3 response tallies at each of the 9 stimulus levels can be reconstructed
exactly, in all three response-set conditions.

## Access (a DIRECT url you have fetched; open, paywalled or Wayback-only)

Open, author-hosted full text (fetched, 367 KB, complete article):
https://www.andywills.info/assets/pdf/2000Wills.pdf

Raw data for Experiment 2 (fetched; 8064 rows, tab-separated, 439 KB; 36 subjects,
3 conditions x 12, columns cond/subj/fixed/phase/trial/catordist/ic1..ic12/resp/rt):
https://www.andywills.info/willslab-dau/cam1/cam1data.txt

Reproduction script (fetched, 105 lines of R, reproduces Figures 4a/4b/4c and both
polynomial regressions; cond 1 = Three-choice, 2 = Two-choice, 3 = Novel-elements):
https://www.andywills.info/willslab-dau/cam1/cam1analysis.R

Data landing page (fetched; also lists cam1stim.tbz stimulus images):
https://www.andywills.info/willslab-dau/cam1/

Metadata and machine-readable abstract (fetched):
https://api.openalex.org/works/doi:10.1080/713755935

Publisher copy is paywalled: https://journals.sagepub.com/doi/pdf/10.1080/713755935
returns HTTP 403 (fetched, blocked). The willslab.co.uk mirrors of the data files are dead
(404); use the andywills.info paths above.

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**Usable now, and the single best asset found in this sweep.** Experiment 2's raw
trial-level data are downloadable, the analysis script documents the condition and
counterbalance coding, and the design is exactly a master-versus-restricted response set
over identical stimuli (three-choice {A,B,C} vs two-choice {B,C}), plus a third condition
that suppresses one alternative's evidence without removing its label. From the raw file
one can build per-subject and pooled 3x3 (and 2x2) response matrices at each of 9 stimulus
levels and test proportional renormalization directly rather than through the authors' q
statistic.

Two caveats. (1) The response-set manipulation is **between subjects**, so any within-subject
odds-ratio test has to be constructed across matched groups. (2) Experiment 1's data do not
appear in the published archive — only Experiment 2 (cam1).

## What the authors concluded, quoted verbatim where possible

From the abstract (ligature-restored from the PDF):

> "Many theories assume that these numbers may be translated into choice probabilities via
> the ratio rule, also known as the choice axiom (Luce, 1959) or the constant-ratio rule
> (Clarke, 1957). ... The ratio rule is shown to be incorrect for these experiments, given
> the assumption that the magnitude terms for each category are univariate functions of the
> number of category-appropriate symbols contained in the presented stimulus. A
> connectionist winner-take-all model of categorical decision (Wills & McLaren, 1997) is
> shown to account for our data given the same assumption. The central feature underlying
> the success of this model is the assumption that categorical decisions are based on a
> Thurstonian choice process (Thurstone, 1927, Case V) whose noise distribution is not
> double exponential in form."

Experiment 2 Discussion:

> "The results of Experiment 2 directly contradict the predictions of the ratio rule acting
> on the output of a magnitude-based model of categorization, within the assumption that
> magnitude is a univariate function of category-appropriate elements."

General Discussion, the central claim — **this is the direct precedent for the project's
thesis**:

> "Our central conclusion is that the ratio rule is an inappropriate theory of categorical
> decision and should be replaced by a system based on the principles of Thurstonian
> choice."

And the sharpened restatement:

> "If the ratio rule is considered in this way then our central conclusion is more properly
> stated as ``the Case V double exponential Thurstonian choice process is an inappropriate
> model of categorical decision, but other Thurstonian choice processes are potentially
> appropriate''."

On what separates the two accounts:

> "However, one might alternatively consider the ratio rule to be a statement that people
> make probabilistic judgements on the basis of deterministic magnitude terms, in contrast
> to the Thurstonian theory that people make deterministic judgements on the basis of
> probabilistic magnitude terms. If considered in this manner, the ratio rule and
> Thurstonian choice are clearly different classes of explanation."

On the scope qualification (worth quoting when pre-empting objections):

> "the ratio rule is incorrect for models that have no process by which information about
> allowed decision alternatives can affect the magnitude terms produced" ... "This
> qualification excludes none of the categorization models cited in this paper from our
> conclusions."

On what does the work, mechanistically:

> "Indeed, the partial success of our simple WTA model suggests that it is the general
> principles of Thurstonian choice, rather than the competitive race itself, that underly
> the success of our full model. Many models employing these general principles are likely
> to be able to explain many of our results (e.g., Ashby & Townsend, 1986), as long as the
> noise distribution employed does not render their predictions indistinguishable from
> those of the ratio rule."

They also note the Yellott (1977) equivalence explicitly:

> "Yellott (1977) proved for situations involving three or more choices that the predictions
> of Thurstone's theory and the ratio rule can be equivalent if and only if the
> distributions employed in Thurstone's theory are double exponential."

And, on the prior literature they are pushing back against (Clarke 1957, Pollack & Decker
1960, Bradley 1954, Hopkins 1954):

> "The problem in accepting such studies as good evidence in support of the ratio rule is
> that, as in the analysis of pair-comparison experiments, no alternative theory is
> considered."
