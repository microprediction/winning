# Wills, Reimers, Stewart, Suret & McLaren 2000 — Tests of the ratio rule in categorization

## Citation

Wills, A.J., Reimers, S., Stewart, N., Suret, M., & McLaren, I.P.L. (2000). Tests of the ratio rule in categorization. *The Quarterly Journal of Experimental Psychology, Section A: Human Experimental Psychology*, 53A(4), 983–1011. doi:10.1080/713755935. PMID 11131824.

Note: the assigned candidate listing gave three authors; the paper has five (Suret and McLaren are also authors). Andy Wills' own publication list omits Suret; the PDF title page includes him.

## Domain and stimuli

Artificial visual prototype-structured categorization, human undergraduates (Cambridge). Each stimulus is a collection of 12 small pictures ("elements") arranged on an invisible 4×3 grid inside a 4.5 × 3.5 cm rectangle. Elements are drawn from a pool of 36 (Experiment 1) or 40 (Experiment 2); for each subject, 12 elements are randomly designated Category A, 12 Category B, 12 Category C (Experiment 2 leaves 4 unallocated as "novel" elements).

Training: 90 observational, labelled stimuli (30 per category), each built from the 12 category-characteristic elements with a 10% per-element substitution rate.

Test (transfer): every target test stimulus holds the Category-a element count **fixed at 4** and varies the other two, containing 4 a-elements, x b-elements and (8 − x) c-elements, x ∈ {0,…,8} — nine stimulus types, 10 exemplars each. Four dummy stimulus types (10 each) disguise the constancy of the a-count. 130 test trials total (90 target + 40 dummy).

Experiment 1: n = 24 (12 per condition), labels not counterbalanced (A always the fixed/disallowed category). Experiment 2: n = 36 (12 per condition), label assignment fully counterbalanced across three sub-groups (Table 1).

## Master and restricted response sets

This is an explicit **full-set vs. sub-set response-set manipulation**, run **between subjects**:

- **Full menu (three-choice condition):** all three responses available. Subjects asked "Is this an A, a B, or a C?" Yields P(a: a,b,c), P(b: a,b,c), P(c: a,b,c).
- **Restricted menu (two-choice condition):** the fixed category's response is **disallowed by the experimenter**. Subjects asked "Is this a B or a C?" Yields P(b: b,c), P(c: b,c) over the *same* stimulus construction.
- **Experiment 2 adds a third condition (novel-elements):** three responses still allowed, but the 4 fixed-category elements in each test stimulus are replaced by 4 elements never seen in training. This lowers the magnitude term for category a without removing the response — a "soft" restriction. Yields P(b: a,b,c)′.

The authors state the design explicitly (p. 986): "In the two-choice decision the same stimuli are presented but one of the responses (A) is disallowed by the experimenter… This procedure is an example of the full-set vs. sub-set manipulation described earlier."

Their test statistic is q = [P(B: B,C) − P(B: A,B,C)] / P(B: A,B,C) (Eq. 4), which under the ratio rule reduces to ν_A/(ν_B + ν_C) (Eq. 6). The qualitative prediction under test: q and P(A: A,B,C) must move in the same direction over any interval of the category-appropriate-element index. Note this is a *derived* qualitative test that leans on an auxiliary assumption (magnitude terms are univariate functions of the element count and are unaffected by which alternatives are allowed) — it is not a raw cell-by-cell CRR renormalization scoring.

## What numbers are printed or deposited

**Printed:** essentially nothing usable. Table 1 is a design table (allocation of category labels to sub-groups in Experiment 2). All results appear in **figures only** — Figure 3a–c (Experiment 1: mean response probability, q, and P(A: A,B,C) vs. number of category-appropriate elements) and Figure 4a–c (Experiment 2, including q′ and P(a: a,b,c)′), plus Figure 6 (WTA model simulations). Best-fit quadratic coefficients are given in the text (e.g. p. 993: P(A) = −0.003c² + 0.021c + 0.29; p. 994: P(A: A,B,C) = −0.03d² + 0.08d + 0.30; p. 998: P(a: a,b,c)′ = −0.003b² + 0.016b + 0.112). No confusion matrices and no tables of choice proportions.

**Deposited — this is the decisive point.** Trial-level raw data for **Experiment 2** are deposited and live as Wills' "Data and Analysis Unit: CAM1" (created 2014 in response to a data request). I downloaded `cam1data.txt`: 8,065 lines (36 subjects × 224 rows), tab-separated with a header, columns:

`cond` (1 = three-choice, 2 = two-choice, 3 = novel-elements), `subj`, `fixed` (the category holding 4 elements, and the response disallowed in the two-choice test phase; 1 = A, 2 = B, 3 = C), `phase` (0 = prototype listing, 1 = training, 2 = test), `trial`, `catordist` (phase 2: values 10–13 flag dummy stimuli; values 1–9 encode the b/c element split conditional on `fixed`), `ic1`–`ic12` (the 12 icons presented, keyed to `cam1stim.tbz` filenames), `resp` (1 = A, 2 = B, 3 = C, −1 = no response required), `rt` (seconds).

Also deposited: `cam1stim.tbz` (stimulus images), `cam1analysis.R` (script reproducing the Experiment 2 analyses), `cam1source.tbz` (BBC BASIC V source), `cam1vid.mp4`. Experiment 1 data are **not** deposited.

Because `cond`, `fixed`, `catordist` and `resp` are all present per trial, full-menu choice shares and restricted-menu choice shares can be recomputed exactly from the deposit for Experiment 2, at any level of aggregation (including per subject).

## Access with a fetched url

- Full text PDF, fetched and converted: https://www.andywills.info/assets/pdf/2000Wills.pdf (367 KB; complete article text obtained)
- Publication list confirming the reprint link: https://www.andywills.info/publications
- Data archive readme, fetched: https://www.andywills.info/willslab-dau/cam1/
- Raw data file, downloaded (HTTP 200, 439,465 bytes): https://www.andywills.info/willslab-dau/cam1/cam1data.txt
- Metadata: https://api.semanticscholar.org/graph/v1/paper/DOI:10.1080/713755935?fields=title,abstract,year,venue,openAccessPdf,externalIds,authors
- OA status (bronze): https://api.unpaywall.org/v2/10.1080/713755935?email=peter.cotton@gsmc.ai
- The publisher PDF at https://journals.sagepub.com/doi/pdf/10.1080/713755935 returned HTTP 403 to me; the author copy above is the same published version.

## Usability verdict

**CRR-TEST-WITH-PRINTED-MATRICES** — but strictly by way of the *deposit*, not the print. The article itself prints no usable numbers (figures only), so on printed content alone this would be CRR-TEST-BUT-NUMBERS-NOT-PRINTED. The live, complete, trial-level Experiment 2 deposit removes that limitation.

The printed-plus-deposited numbers **do suffice** to score a full-menu-calibrated prediction against CRR on a restricted menu, for Experiment 2. Concretely: calibrate on the `cond == 1` (three-choice) shares for each of the nine b/c element splits, renormalize onto {b, c} to get the CRR prediction, and score against the observed `cond == 2` (two-choice) shares. The `cond == 3` novel-elements condition supplies a second, softer restriction.

Caveats a scorer must accept:
1. **Between-subjects.** The full-menu and restricted-menu shares come from different people (12 each), so any group difference is confounded with the menu manipulation. The authors acknowledge this and explain why a within-subjects design is problematic for q (P(b: a,b,c) can be 0, making q undefined) — General Discussion, pp. 1006–1007.
2. Aggregate-level analysis in the paper; the deposit permits subject-level, but with only 12 subjects per cell.
3. This is **not** an N-alternative identification confusion matrix. It is a 3-category classification over a graded stimulus continuum, with the restricted menu formed by deleting one category response. That is a clean CRR/IIA test but has a different geometry from a Clarke-style master confusion matrix.
4. Only 3 responses in the full menu and 2 in the restricted menu — a small menu, so Thurstone-vs-Luce discrimination power per cell is limited (Yellott 1977 equivalence is only broken for n ≥ 3, which this just satisfies).

## Conclusion about CRR quoted verbatim

"Our central conclusion is that the ratio rule is an inappropriate theory of categorical decision and should be replaced by a system based on the principles of Thurstonian choice." (p. 1008)

The authors immediately qualify this: "If the ratio rule is considered in this way then our central conclusion is more properly stated as ``the Case V double exponential Thurstonian choice process is an inappropriate model of categorical decision, but other Thurstonian choice processes are potentially appropriate''." (p. 1008)

And a second qualification, on p. 1007: "This argument may be seen as a qualification of our conclusions, which may thus be stated more fully as ``the ratio rule is incorrect for models that have no process by which information about allowed decision alternatives can affect the magnitude terms produced''. This qualification excludes none of the categorization models cited in this paper from our conclusions."

From the abstract (p. 983): "The ratio rule is shown to be incorrect for these experiments, given the assumption that the magnitude terms for each category are univariate functions of the number of category-appropriate symbols contained in the presented stimulus. A connectionist winner-take-all model of categorical decision (Wills & McLaren, 1997) is shown to account for our data given the same assumption. The central feature underlying the success of this model is the assumption that categorical decisions are based on a Thurstonian choice process (Thurstone, 1927, Case V) whose noise distribution is not double exponential in form."
