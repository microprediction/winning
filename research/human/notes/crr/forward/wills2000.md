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

## Whether this is prior art for a parameter-free Gaussian out-of-sample test

Short answer: **no — it is partial prior art for the negative half only.** Wills et al. refute the ratio rule qualitatively, and they advocate Thurstonian choice in the abstract and conclusion, but the positive model they actually run is (a) **rectangular**-noise, not Gaussian; (b) **four-free-parameter and fitted per condition**, with the restricted menu given its own parameter value; (c) **never scored numerically against a proportional-renormalization benchmark**; and (d) calibrated from an assumed linear magnitude function, **not** from inverting observed full-menu shares. Point-by-point below.

### 1. The WTA model's noise distribution is RECTANGULAR, not Gaussian

Decisive. The full WTA model's noise is explicitly uniform:

> "The term r_i,c in this equation is the value of the noisy input produced by the magnitude term ν_i and presented to unit i on update c. In the simulations that follow the noise added to ν_i ranges from +N to −N, has a mean of zero, and has a **rectangular distribution** (i.e., all values from +N to −N are equally likely). Superimposed on this noise function is the constraint that r_i,c cannot exceed one or fall below zero." (p. 1002, emphasis added)

The Gaussian is invoked in only two places, neither of which is the model that carries the paper's argument:

- **In the Introduction, as a definition of Thurstone's theory, not as their model:** "If the magnitude terms are assumed to have a Gaussian distribution then this alternative theory corresponds to Thurstone's (1927) theory of judgement, with our term psychological magnitude basically corresponding to Thurstone's term discriminal process." (p. 985)
- **As a one-sentence aside on the stripped-down "simple-WTA" model, with no figure and no numbers:** having described simple-WTA predictions computed from "terms with a rectangular distribution with a width of 0.7 and means determined by Equation 13", they add: "Employing Gaussian distributions with a standard deviation of 0.28 (which have a rectangular equivalent with a width of 0.7) produces comparable results." (p. 1005)

That aside is the closest the paper comes to the project's model, and it is a single unillustrated, unquantified remark about a model they have just described as fitting **worse**: "The correspondence between predictions and data is by no means as good for this simplified model as it is for the full model (see Figures 6d–6f)." (p. 1005) The simple-WTA also **fails outright** on one of the four target functions: "However, the presented simulation of the simple-WTA model does not correctly predict the trend in the q′ statistic." (p. 1005)

Note also footnote 3 (p. 1009), where they go out of their way to *detach* Gaussianity from Thurstone: "Considering the ratio rule in this way assumes that Gaussian distributions are not a defining property of Thurstone's theory."

So on the crux the peer identified: they argued **the negative** ("not double exponential"), and ran a rectangular-noise positive model. They did not run the Gaussian-beats-renormalization comparison.

### 2. The WTA model is fitted, has four free parameters, and gets a DIFFERENT parameter value for the restricted menu

The paper states the parameter count itself:

> "The WTA model is a relatively complex system with four free parameters (E, D, N, and S). The ratio rule, in contrast, has no free parameters—its predictions are entirely determined by the magnitude terms that it is presented with. This contrast in complexity raises two related questions. First, is it simply increased complexity and, in particular, the presence of more parameters, that permits the WTA model to successfully account for our data?" (p. 1005)

**The decisive detail — the decision threshold S is set per condition, i.e. the restricted menu receives its own fitted value:**

> "In the current simulation, S is set to **0.18 for the two-choice condition, 0.65 for the three-choice condition, and 0.72 for the novel-elements condition**. Employing a different value of S for each condition is in line with previous applications of the model where we have assumed that both the type of decision (two-choice vs. three-choice here), and the presence of novel elements in test stimuli, affects the value of S (Wills, 1998; Wills & McLaren, 1997)." (p. 1002, emphasis added)

No fitting procedure, search, or goodness-of-fit criterion is stated for these three values. The other three parameters are carried over from earlier work rather than fitted here: "The remaining parameters, E, D and N, are set to 0.2, 0.1, and 1.1, respectively. These values are the same as those employed by Wills and McLaren (1997) in the simulation of their experiments, and by Wills (1998) in the simulation of the experiments presented in Jones et al. (1998)." (p. 1002)

Because S differs between the three-choice and two-choice conditions, **the WTA account of the two-choice condition is not an out-of-sample prediction from full-menu-calibrated quantities.** It is a simulation with a menu-specific parameter. Indeed the authors later lean on exactly this freedom to rescue the q′ trend: "we believe that it is the adoption of different decision thresholds, for the three-choice and novel-elements conditions in the full model that allows it to correctly predict the q′ data. Were S to take the same value in both conditions, the probability of choosing, say, the Category b response for a stimulus with x Category b elements would always be greater in the novel-elements condition than in the three-choice condition (and hence q′ would always be positive)." (pp. 1005–1006)

**The magnitude terms are also not derived from observed full-menu shares.** They are assumed linear and taken from a network model at a chosen learning rate:

> "We will assume for the purposes of this simulation that these magnitude terms are linear functions of number of category-appropriate elements. Each category is assumed to have the same magnitude function, which, in the current simulations, takes the form ν_i = 0.047c_i + 0.012" (p. 1002)

> "Equation 13 specifies this function for the network model presented in Wills and McLaren (1997) when the model's learning-rate parameter is set to 0.0025. This value is of the same order of magnitude as learning rates that we have employed previously in simulating experiments of this sort." (p. 1003)

The simulation is Monte Carlo, 50,000 draws per stimulus type per condition (p. 1003), and the restricted menu is implemented by clamping the disallowed unit: "In the two-choice condition of our experiment, subjects were not allowed to make Category a responses. In our WTA model this was simulated by fixing the output activation of the Category a unit (o_a) at zero… The assumption made in doing this is that only allowed responses compete for the right to produce a response." (p. 1003)

**No fit statistic is reported anywhere for the WTA simulation** — the assessment is visual: "the data and predictions correspond fairly closely, although it may be noted that the values predicted for q and P(a: a,b,c)′ are slightly lower than those observed. Nevertheless, our simulation demonstrates that the WTA model is capable of predicting the major trends observed in our experiment." (p. 1003)

For fairness, the simple-WTA *is* described as parameter-free, and this is the sentence closest to the project's framing — but note it still requires a chosen distribution width, and it fits worse and fails on q′: "The probability with which this simple-WTA system picks each alternative is entirely determined by the means and distributions of the magnitude terms. Like the ratio rule, the simple-WTA system has no free parameters, although more information about the magnitude terms is required to derive predictions from the simple-WTA system." (p. 1005)

### 3. No cell-level or share-level renormalization prediction is ever computed or scored

The entire empirical test is carried by the derived statistic **q** and its **qualitative directional/shape** prediction. There is no point where the paper writes down "CRR predicts P(b: b,c) = X, observed = Y, discrepancy = Z", and no discrepancy measure, χ², RMSE, or deviation criterion against a renormalized prediction appears anywhere.

What is actually reported as the test: best-fitting quadratics to q, q′, P(a: a,b,c) and P(a: a,b,c)′ against the category-appropriate-element index, with F- and t-tests on the regression terms, and the conclusion that the fitted curves have *opposite shape* (U vs. inverted-U) where the ratio rule requires the same direction of change. Representative fitted lines: "The equation of the line is P(A) = −0.003c² + 0.021c + 0.29" (p. 993); "P(A: A,B,C) = −0.03d² + 0.08d + 0.30. This function, shown as a solid line in Figure 3c, suggests a downward trend but did not significantly fit the data, F(2, 2) = 2.9, p > .25." (p. 994); "P(a: a,b,c)′ = −0.003b² + 0.016b + 0.112" (p. 998). The Experiment 2 verdict is stated in exactly these shape terms: "the fact that the best-fitting quadratics for the q and P(a: a,b,c) functions are of opposite shape (U vs. inverted-U shape) is contrary to the predictions of the ratio rule." (p. 999)

**A gap worth recording, because it bears directly on the boundary.** Their own Equations 3 and 6 imply an *exact, parameter-free* CRR point prediction that they never form or score. Since q = ν_A/(ν_B + ν_C) (Eq. 6) and P(A: A,B,C) = ν_A/(ν_A + ν_B + ν_C) (Eq. 3), it follows algebraically that CRR requires

  q = P(A: A,B,C) / [1 − P(A: A,B,C)]

with the right-hand side computable from the **three-choice condition alone**. Equivalently, CRR predicts P(b: b,c) = P(b: a,b,c) / [P(b: a,b,c) + P(c: a,b,c)] — straight proportional renormalization. The paper uses only the weak monotone corollary (both functions must move the same way) and never evaluates the exact identity numerically. So the *specific* comparison the project performs — a scored, cell-level, parameter-free renormalization benchmark — is **not** in this paper, even though the data to compute it are in the deposit.

### 4. None of "parameter-free", "out of sample", "cross-validation" or "held-out" appears

Case-insensitive searches over the full converted text: **"parameter-free" 0 hits, "parameter free" 0, "out of sample" 0, "out-of-sample" 0, "cross-validat*" 0, "held-out" 0, "held out" 0.** ("free parameter" occurs 5 times, all in the p. 1005 passage about the WTA model *having* four of them.)

"Prediction" is used freely, but in two senses that must not be conflated: predictions *of a theory* about the qualitative form of a function, and — for the ratio rule specifically, when describing the **prior** literature — genuine sub-set forecasting. The latter appears only in the Introduction's summary of Clarke: "The full-set choice probabilities were used to derive predictions, via the ratio rule, for the sub-set choice probabilities. These predictions were then compared with the observed choice probabilities." (p. 986) They describe that as the design **Clarke (1957)** ran, and they explicitly criticise it for lacking a comparison theory: "The problem in accepting such studies as good evidence in support of the ratio rule is that, as in the analysis of pair-comparison experiments, no alternative theory is considered." (p. 986) They do not themselves run a Clarke-style scored forecast.

### 5. Yellott (1977) is cited, and the pairs-vs-three-or-more divergence is stated explicitly

Yes — 4 occurrences (pp. 985, 986, 1008, and the reference list p. 1011: "Yellott, J.I., Jr. (1977). The relationship between Luce's choice axiom, Thurstone's theory of comparative judgement…"). The theorem is stated precisely, including the n ≥ 3 restriction:

> "As has been noted previously (e.g., Luce, 1959, p. 56) the ratio rule and Thurstone's theory can often make very similar predictions. However, Yellott (1977) proved for situations involving three or more choices that the predictions of Thurstone's theory and the ratio rule can be equivalent if and only if the distributions employed in Thurstone's theory are double exponential. For a two-choice situation there are distributions other than the double exponential that allow equivalence (e.g., an exponential distribution)." (p. 985)

> "Yellott's demonstrations were for Case V of Thurstone's theory, which assumes that all distributions have the same variance." (p. 986)

And in the General Discussion, used to recast their own conclusion:

> "However, the ratio rule and Thurstonian choice need not necessarily be considered as different classes of explanation. As discussed earlier, Yellott (1977) demonstrated that the predictions of the ratio rule are equivalent to a Case V Thurstonian choice process with double exponential noise distributions. As such, the ratio rule may be considered as a description of one member of the set of Thurstonian choice processes." (p. 1008)

So the project cannot claim novelty for the Yellott framing — that is squarely established here. Note the direction of the paper's use of it, though: Yellott licenses them to say Gaussian-Case-V *could* differ from the ratio rule at n ≥ 3, which motivates the search for a non-double-exponential model. They then pick a rectangular one.

### 6. Shown vs. argued — fullest verbatim statements

**Claimed as SHOWN (always hedged by an auxiliary assumption):**

> "Any theory of learning and memory whose output is a set of magnitude terms must specify how these terms translate into testable predictions. Where those predictions concern response probabilities, it is commonly assumed that the ratio rule provides the appropriate translation. **With certain qualifications, this assumption has been shown to be incorrect** for the categorization experiments presented in this paper." (p. 1006)

> "The ratio rule is shown to be incorrect for these experiments, **given the assumption that** the magnitude terms for each category are univariate functions of the number of category-appropriate symbols contained in the presented stimulus." (Abstract, p. 983)

> "The results of Experiment 2 directly contradict the predictions of the ratio rule acting on the output of a magnitude-based model of categorization, **within the assumption that** magnitude is a univariate function of category-appropriate elements." (p. 999)

> "Hence, for models producing linear or monotonically accelerating magnitude functions, the ratio rule is an inappropriate theory of the decision process in categorization (**within the assumptions made**)." (p. 995)

> "Nevertheless, our simulation demonstrates that the WTA model is capable of predicting the major trends observed in our experiment. We have already determined that (**within certain assumptions**) the ratio rule is unable to do so." (p. 1003)

**Claimed as ARGUED / advocated / conditional:**

> "Our central conclusion is that the ratio rule is an inappropriate theory of categorical decision and should be replaced by a system based on the principles of Thurstonian choice." (p. 1008) — an *ought*, and immediately restated more narrowly: "If the ratio rule is considered in this way then our central conclusion is more properly stated as ``the Case V double exponential Thurstonian choice process is an inappropriate model of categorical decision, but other Thurstonian choice processes are potentially appropriate''." (p. 1008)

> "This argument may be seen as a qualification of our conclusions, which may thus be stated more fully as ``the ratio rule is incorrect for models that have no process by which information about allowed decision alternatives can affect the magnitude terms produced''. This qualification excludes none of the categorization models cited in this paper from our conclusions." (p. 1007)

> "Conversely, if the assumptions we have made in coming to our conclusions can be shown to be invalid then the ratio rule is not necessarily incorrect." (p. 1006)

They explicitly decline to claim their WTA model is uniquely supported, and concede the class is wide open:

> "By using our WTA model to simulate the data presented in this paper we do not intend to imply that it is the only model of its class that has the potential to explain our results. Indeed, the partial success of our simple WTA model suggests that it is the general principles of Thurstonian choice, rather than the competitive race itself, that underly the success of our full model. Many models employing these general principles are likely to be able to explain many of our results (e.g., Ashby & Townsend, 1986), **as long as the noise distribution employed does not render their predictions indistinguishable from those of the ratio rule**." (p. 1009)

They also flag the aggregate-data limitation themselves: "we have estimated the shape of our four functions, q, q′, P(a: a,b,c), P(a: a,b,c)′, from mean data rather than from the data of individual subjects. On this basis, one could argue that although we have demonstrated the ratio rule to be incorrect for average responses, it may actually be correct for individuals." (p. 1006)

### Boundary summary — how to characterise this paper if it is cited as prior art

| Project's method | Wills et al. 2000 |
|---|---|
| Gaussian (Thurstone Case V) noise | **Rectangular** noise in the model that carries the argument; Gaussian appears only as a one-line unquantified aside on the worse-fitting simple-WTA (p. 1005) |
| Parameter-free; nothing fitted beyond inverting full-menu shares | Full WTA has **four free parameters**, self-described (p. 1005); magnitude terms assumed linear from a chosen learning rate (Eqs. 13, p. 1002), **not** inverted from observed shares |
| Nothing ever fitted to the restricted menu | **S is set per condition** — 0.18 two-choice vs 0.65 three-choice (p. 1002); the q′ account explicitly depends on this freedom (pp. 1005–1006) |
| Restricted-menu shares predicted out of sample | No out-of-sample language anywhere (0 hits); the two-choice condition is *simulated*, with its own parameter |
| Scored against proportional renormalization | **Never computed.** Test is the qualitative same-direction/shape prediction on q and P(a: a,b,c); no discrepancy statistic against a renormalized prediction |
| Yellott n ≥ 3 framing | **Already established here** (pp. 985, 986, 1008) — claim no novelty for this |

Fair characterisation for circulation: Wills et al. (2000) is the **closest prior art on the refutation side** — it is a genuine within-paper full-menu/restricted-menu manipulation that concludes against the ratio rule and in favour of Thurstonian choice, and it already owns the Yellott framing. It is **not** prior art for a parameter-free Gaussian out-of-sample test scored against proportional renormalization: the noise is rectangular, the positive model is fitted with a menu-specific threshold, and the renormalization benchmark is never numerically computed. The exact parameter-free CRR identity implied by their own Equations 3 and 6 — q = P(A)/[1 − P(A)] — is left unscored, and can be evaluated directly from the CAM1 deposit.
