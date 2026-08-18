# Yeon & Rahnev (2020) — multi-alternative vs. two-alternative perceptual choice

Downloaded 2026-08-17. Everything in this directory is public.

## Citation

> Yeon, J., & Rahnev, D. (2020). The suboptimality of perceptual decision making with
> multiple alternatives. *Nature Communications*, **11**, 3857.
> https://doi.org/10.1038/s41467-020-17661-z

- DOI: `10.1038/s41467-020-17661-z`
- Published 2020-07-31. Vol 11, issue 1, article number 3857. ISSN 2041-1723.
- PMID 32737317, PMCID PMC7395091 (open access full text:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7395091/)
- Preprint (earlier title "The nature of the perceptual representation for decision
  making"): bioRxiv https://doi.org/10.1101/537068 (v1 2019-01-31, v2 2020-03-25)

Data citation given by the authors (ref. 50 of the paper):

> Yeon, J. & Rahnev, D. *On the nature of the perceptual representation at the decision
> stage*. OSF (2020). https://doi.org/10.17605/OSF.IO/D2B9V

## Identifiers and download URLs (all verified to resolve 2026-08-17)

| What | URL | HTTP |
|---|---|---|
| Article | https://doi.org/10.1038/s41467-020-17661-z | 200 |
| Article (direct) | https://www.nature.com/articles/s41467-020-17661-z | 200 (redirects via idp.nature.com; adds `?error=cookies_not_supported`) |
| Open-access full text | https://pmc.ncbi.nlm.nih.gov/articles/PMC7395091/ | 200 |
| **OSF project (data + analysis code)** | https://osf.io/d2b9v/ — GUID `d2b9v`, DOI `10.17605/OSF.IO/D2B9V` | 200 |
| OSF bulk zip (GET only) | https://files.osf.io/v1/resources/d2b9v/providers/osfstorage/?zip= | 200 on GET (HEAD returns 501 — use `curl -L -o`) |
| OSF preregistration (Exps 2 & 3) | https://osf.io/dr89k/ | 200 |
| Confidence Database (Exp 1 only) | https://osf.io/s46pr/ | 200 |
| Confidence DB — Exp 1 data | https://osf.io/download/p5du8/ (`data_Yeon_2019.csv`) | 200 |
| Confidence DB — Exp 1 readme | https://osf.io/download/mezhf/ (`readme_Yeon_2019.txt`) | 200 |
| Nature Supplementary Information | https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-17661-z/MediaObjects/41467_2020_17661_MOESM1_ESM.docx | 200 |
| Nature Reporting Summary | https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-17661-z/MediaObjects/41467_2020_17661_MOESM2_ESM.pdf | 200 |
| **Nature Source Data** | https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-17661-z/MediaObjects/41467_2020_17661_MOESM3_ESM.xlsx | 200 |

Licence on the OSF project: CC-BY 4.0, copyright holder Jiwon Yeon, 2020.

Dead link: the Confidence Database readme points at
`https://github.com/wiseriver531/Discrete-representation` for materials — **404** as of
2026-08-17.

## What is in this directory

```
SOURCE.md                      this file
build_tidy.py                  script that produced tidy/ from osf/d2b9v (needs numpy + scipy)
osf/
  d2b9v_osfstorage.zip         8,002,755 B — bulk download of the whole OSF project
  d2b9v/                       extracted, 318 files, 8.4 MB
  MANIFEST_osf_d2b9v.tsv       318 rows: path, size, materialized_path, per-file download URL
nature_source_data/
  41467_2020_17661_MOESM1_ESM.docx   813,929 B  Supplementary Information (Supp. Methods/Notes/Figures)
  41467_2020_17661_MOESM2_ESM.pdf  2,255,815 B  Nature Research Reporting Summary
  41467_2020_17661_MOESM3_ESM.xlsx    40,644 B  Source Data
confidence_database/
  data_Yeon_2019.csv           1,092,778 B  Experiment 1, standardised Confidence Database format
  readme_Yeon_2019.txt             2,849 B  variable documentation for the above
tidy/                          derived CSVs, see below (not from the authors)
```

OSF project layout, identical under each of `Experiment 1/` … `Experiment 4/`:

```
Experiment N/
  data/subject_responses/raw responses/*.mat   trial-level, one file per subject(-session)
  data/subject_responses/dataForModeling.mat   aggregate count matrices actually fit
  data/fitting results/{simple,extended,attention_extended}/*.mat   fitted params + predictions
  codes/analysis/organize behavioral responses/*.m   raw -> dataForModeling
  codes/analysis/fitting_*/*.m                       simulated-annealing MLE fitting
  codes/analysis/print results/{Run_code.m,AICanalysis.m}
```

File counts / sizes: Exp 1 = 89 files / 1.3 MB; Exp 2 = 89 / 2.4 MB; Exp 3 = 88 / 2.6 MB;
Exp 4 = 52 / 2.1 MB.

## The raw MATLAB files

These are nested MATLAB structs, not tables, so "column names" below means struct fields.
Read with `scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)`.

### Experiment 1 — 4 colours → 2 colours, 32 subjects

`Experiment 1/data/subject_responses/raw responses/results_s{1..32}.mat`, 28,901–35,126 B each.

Single variable `p`. Trial data live in the 6×3 cell array `p.data{run, block}` (6 runs ×
3 blocks × 35 trials = 630 trials/subject). Each block struct has:

| field | meaning |
|---|---|
| `condition` | **1** = 4-alternative (full menu); **2** = 2-alternative, menu revealed *after* stimulus offset; **3** = 2-alternative, menu announced *in advance* ("advance warning"). 210 trials per condition. |
| `correctColor` | 1×35, the dominant colour (the "stimulus"), 1–4 |
| `wrongColor` | 1×35, **the offered non-dominant alternative** — present only for conditions 2 and 3 |
| `response` | 1×35, colour chosen, 1–4 |
| `correct` | 1×35, 0/1 |
| `confidence` | 1×35, 1–4 |
| `rt` | 35×2, [decision RT, confidence RT] in s |
| `presentation_time` | 35×4, GetSecs timestamps |

`p.colorPairs` (6×2) and `p.colorOrder` schedule which pair is used in condition-3 blocks.
Colour codes, from the Confidence Database readme: **1 = red, 2 = green, 3 = blue,
4 = white**. (The paper's Methods prose lists the colours in a different order; the readme
is the authoritative key.)

Aggregate: `Experiment 1/data/subject_responses/dataForModeling.mat`, 1,608 B, one struct `data`:
- `respPattern_cond1` — `uint8[32, 4, 4]` = counts indexed `[subject, dominant_colour, response]`. This is the full-menu response distribution.
- `respPattern_cond2` — `uint8[32, 4, 4, 2]` = counts indexed `[subject, dominant_colour, alternative_colour, 1=correct/2=wrong]`. **This is the restricted-menu distribution, keyed by menu identity.**
- `accuracy` — `float[32, 3]`, mean accuracy per condition.
- Note the condition-3 (advance-warning) counts are computed by the cleanup script but **not** saved into `dataForModeling.mat`; recover them from the raw files.

### Experiment 2 — 6 symbols → 2 symbols, 10 subjects × 3 sessions

`Experiment 2/data/subject_responses/raw responses/sub{1..10}_{1..3}.mat`, 61,824–78,368 B each.
1000 trials/session, 3000/subject.

Single variable `p`; trials in the 1×20 cell `p.main{block}` (5 runs × 4 blocks × 50 trials):

| field | meaning |
|---|---|
| `NchoiceOption` | 6 (full menu) or 2 (restricted menu); blocks alternate, schedule mirrored in `p.optionN2` |
| `targetOrder` | 1×50, dominant symbol 1–6 |
| `pairOrder` | 1×50, **the offered non-dominant alternative** — present only in `NchoiceOption == 2` blocks |
| `response` | 50×1, symbol chosen 1–6 |
| `correct` | 50×1, 0/1 |
| `stimOrders` | 50×49, the literal 7×7 display (which symbol at each of 49 grid cells) |
| `rt`, `startTime`, `presentation_time`, `stimDuration` | timing |

Symbol codes come from `p.stimSet` = `{'?', '#', '$', '%', '+', '>'}` (index = code).
No confidence ratings in Exp 2.

Aggregate: `Experiment 2/data/subject_responses/dataForModeling.mat`, 1,281 B:
- `respPattern_cond1` — `uint8[10, 6, 6]` `[subject, dominant_symbol, response]` (full menu).
- `respPattern_cond2` — `uint8[10, 6, 6, 2]` `[subject, dominant_symbol, alternative_symbol, correct/wrong]` (restricted menu, keyed by menu identity).
- `acc` — `float[10, 2]`.

### Experiment 3 — 6 symbols, first answer then second answer, 10 subjects × 3 sessions

`Experiment 3/data/subject_responses/raw responses/sub{1..10}_{1..3}.mat`, 74,241–85,955 B each.
No 2-alternative condition; instead the menu for the second answer is the 5 symbols other
than the first answer. Trials in `p.main{block}`:

| field | meaning |
|---|---|
| `targetOrder` | 1×50, dominant symbol 1–6 |
| `response` | 50×2, [first answer, second answer]; second = **0** when no second answer was requested |
| `correct` | 50×2, [0/1, 0/1 or **99** = no second answer] |
| `stimOrders` | 50×49, the 7×7 display |
| `rt` | 50×2; second column 0 when no second answer |
| `chance2_count` | 1×6, per-block bookkeeping of second-chance trials |

Aggregate: `Experiment 3/data/subject_responses/dataForModeling.mat`, 1,901 B:
- `respPattern_cond1` — `uint16[10, 6, 6]` `[subject, dominant_symbol, first_answer]`, sums to 30,000.
- `respPattern_cond2` — `uint8[10, 6, 6, 6]` `[subject, dominant_symbol, first_answer, second_answer]`, sums to 5,095. **The restricted menu is identified: it is all six symbols minus `first_answer`.**
- `acc` — `float[10, 2]`.

### Experiment 4 — 3 motion directions → 2 directions, 11 subjects × 3 sessions

`Experiment 4/data/subject_responses/raw responses/sub{1..11}_{1..3}.mat`, 50,176–58,021 B each.
1000 trials/session. Single variable `p`, but here everything is a rectangular array
indexed `[run(5), block(4), trial(50)]`:

| field | meaning |
|---|---|
| `limitChoices` | 2×4; row = `1` for odd runs, `2` for even runs; value `1` = 2-alternative block, `0` = 3-alternative block |
| `correctAnswer` | 5×4×50, the on-screen label (1/2/3) of the dominant direction |
| `responses` | 5×4×50, label pressed |
| `correct` | 5×4×50, 0/1 |
| `choices` | 5×4×50×3 — **`[:,:,:,1]` = dominant label, `[:,:,:,2]` = the offered alternative, `[:,:,:,3]` = the excluded direction (MATLAB 1-based).** ⚠ present in only **14 of 33** session files |
| `direction_used` | 5×4×50×3, the three motion directions in radians (dominant, +120°, +240°) |
| `proportions` | 1×3, dot proportions, e.g. `[0.55 0.22 0.23]`; per-subject thresholded |
| `response_time` | 5×4×50 |

Screen labels 1/2/3 are **re-randomised on every trial**, so labels are not stable
direction identities.

Aggregate: `Experiment 4/data/subject_responses/dataForModeling.mat`, 231,008 B. Struct
`data` with `proportion_used[11]`, `c3.{correct,correct_answer,resp,rt}[1500,11]` and
`c2.{correct,correct_answer,resp,rt}[1500,11]` plus `c2.wrong_answer[3000,11]`.
**Do not use `c2.wrong_answer`** — see caveats.

## Nature Source Data — `41467_2020_17661_MOESM3_ESM.xlsx` (40,644 B)

8 sheets: `Figure 4`, `Figure 5`, `Figure 6`, `Figure 7`, `Sup. Figure 1`,
`Sup. Figure 2`, `Sup. Figure 5`, `Sup. Figure 6`. Each sheet is a stack of small labelled
blocks, one per panel; rows are series (`Observed`, `Population model`, `Summary model`,
`2-Highest model`, …) and columns are `data1 … dataN`, one per subject. Example, sheet
`Figure 4`, panel `Figure 4a`:

```
Figure 4a
              data1              data2              data3   ...
Observed      0.795238095238095  0.819047619047619  0.614285714285714
Population model 0.846340833333333 0.9013875       0.766108333333333
Summary model 0.818215          0.860563333333333  0.709601666666667
```

This is **per-subject summary accuracy and AIC differences only** — no trial-level data and
no menu identities. Everything trial-level is on OSF.

## Confidence Database version of Experiment 1

`confidence_database/data_Yeon_2019.csv`, 1,092,778 B, 20,160 data rows (32 subjects × 630
trials), 9 columns:

`Subj_idx, Stimulus, Response, Confidence, RT_dec, RT_conf, Version, Condition, Wrong_option`

```
1,4,4,1,2.41473131199928,4.31905314199958,1,2,1
1,2,2,4,1.567965792,1.95188435599994,1,2,3
1,2,2,4,1.48905611400005,1.92095650700048,1,2,4
```

- `Stimulus` = dominant colour, `Response` = colour chosen (1 = red, 2 = green, 3 = blue, 4 = white).
- `Condition` ∈ {1, 2, 3} exactly as above, 6,720 rows each.
- `Wrong_option` = **the offered alternative**; `NaN` on all 6,720 condition-1 rows, and a
  colour 1–4 on every condition-2 and condition-3 row. The readme states this explicitly:
  *"The data shows the wrong options that were presented for the second and the third
  condition in the column 'Wrong_option'. The same column for the first condition is
  entered with 'NaN'."*
- `Version` 1 = discrete 1–4 confidence, 2 = continuous 0–100 %.
- Row order matches the raw `.mat` files trial-for-trial. This CSV covers Experiment 1 only.

## Derived tidy CSVs (`tidy/`, produced by `build_tidy.py`)

Not from the authors. Trial-level tables (empty string = not applicable) plus long-form
versions of the authors' own count matrices.

| file | bytes | data rows | columns |
|---|---|---|---|
| `exp1_trials.csv` | 724,156 | 20,160 | subject, run, block, condition, trial_in_block, dominant_color, alternative_color, response, correct, confidence, rt_choice, rt_confidence |
| `exp2_trials.csv` | 849,550 | 30,000 | subject, session, run, block, trial_in_block, n_options, dominant_symbol, alternative_symbol, response, correct, rt |
| `exp3_trials.csv` | 949,115 | 30,000 | subject, session, run, block, trial_in_block, dominant_symbol, response1, correct1, response2, correct2, rt1, rt2 |
| `exp4_trials.csv` | 1,891,818 | 33,000 | subject, session, run, block, trial_in_block, n_options, dominant_label, alternative_label, excluded_label, response, correct, rt, prop_dominant_dots, dir1_deg, dir2_deg, dir3_deg |
| `exp1_full_menu_counts.csv` | 5,205 | 512 | subject, dominant_color, response, n |
| `exp1_pair_menu_counts.csv` | 4,899 | 384 | subject, dominant_color, alternative_color, n_correct, n_wrong |
| `exp1_pair_menu_advance_counts.csv` | 4,925 | 384 | same columns, but for condition 3 (menu announced *in advance*); 6,720 trials, pooled accuracy 0.8504 — the authors never saved or analysed this arm |
| `exp2_full_menu_counts.csv` | 3,689 | 360 | subject, dominant_symbol, response, n |
| `exp2_pair_menu_counts.csv` | 3,908 | 300 | subject, dominant_symbol, alternative_symbol, n_correct, n_wrong |
| `exp3_full_menu_counts.csv` | 3,743 | 360 | subject, dominant_symbol, response1, n |
| `exp3_second_answer_counts.csv` | 13,927 | 1,244 | subject, dominant_symbol, response1, response2, n (zero cells omitted) |

Validation — group mean accuracies recomputed from `tidy/` against the published values:

| quantity | recomputed | paper |
|---|---|---|
| Exp 1, 4-alternative | 0.6918 | 69.2 % |
| Exp 1, 2-alternative | 0.7802 | 78 % |
| Exp 1, advance-warning (cond 3, unreported) | 0.8504 | — |
| Exp 2, 6-alternative | 0.5051 | 50.5 % |
| Exp 2, 2-alternative | 0.7163 | 71.6 % |
| Exp 3, first answer | 0.5073 | 50.7 % |
| Exp 3, second answer | 0.2957 | 29.6 % |
| Exp 4, 3-alternative | 0.7741 | 77.4 % |
| Exp 4, 2-alternative | 0.8374 | 83.7 % |

`exp1_pair_menu_counts.csv` reproduces all 384 cells of the authors'
`respPattern_cond2` exactly.

## Restricted-menu identity — is it recorded?

Yes for Experiments 1, 2 and 3; **partially** for Experiment 4.

| Exp | full menu | restricted menu | menu identity recorded? |
|---|---|---|---|
| 1 | 4 colours | 2 colours | **Yes**, `wrongColor` per trial (raw), `respPattern_cond2[·, dom, alt, ·]` (aggregate), `Wrong_option` (Confidence DB). Also for the advance-warning condition 3. |
| 2 | 6 symbols | 2 symbols | **Yes**, `pairOrder` per trial; `respPattern_cond2[·, dom, alt, ·]`. |
| 3 | 6 symbols | 5 symbols (all but the first answer) | **Yes** by construction — the menu is determined by `response(:,1)`, which is recorded; `respPattern_cond2[·, dom, resp1, resp2]` is the full joint. |
| 4 | 3 directions | 2 directions | **Only for 5 of 11 subjects.** `p.choices` exists in 14 of 33 session files: subjects 8, 9, 10, 11 (all 3 sessions) and subject 7 (sessions 2 and 3). Subjects 1–6 and `sub7_1` have no record of which of the two non-dominant directions was offered. 1,500 of the 16,500 2-alternative trials are recoverable per those subjects; totals: sub1–6 = 0/1500 each, sub7 = 1000/1500, sub8–11 = 1500/1500 each. |

Important structural point: the restricted menu is **never an arbitrary pair**. In
Experiments 1, 2 and 4 the two-alternative menu is always {dominant, one randomly chosen
non-dominant}, so the correct answer is always present and no non-dominant/non-dominant
pairs were ever offered. Menus of the form {nondominant_i, nondominant_j} do not exist in
this dataset.

Mitigating the Exp 4 gap: the two non-dominant directions are exactly symmetric (±120° from
the dominant direction) and were modelled with a single shared μ, so which one was offered
carries no design information beyond "a non-dominant". Combined with per-trial
re-randomisation of the 1/2/3 labels, the missing `choices` field costs little for Exp 4
specifically — but if you need labelled menus, only subjects 7(sessions 2–3) and 8–11 are usable.

## Caveats

1. **`Experiment 4/.../dataForModeling.mat` `c2.wrong_answer` is misaligned.** The cleanup
   script (`behavioral_response_cleanup.m`) allocates `wrong_answer_2c = NaN(1,3000)` and
   writes into it using the *global* trial index (0–3000, covering both conditions), while
   `c2.correct`, `c2.resp`, `c2.correct_answer` and `c2.rt` are *compacted* 1500-element
   vectors containing only the 2-alternative trials. So `c2.wrong_answer[i]` does not
   correspond to `c2.resp[i]`. NaN counts per subject: 3000, 3000, 3000, 3000, 3000, 3000,
   2000, 1500, 1500, 1500, 1500. Recompute from the raw session files instead
   (`build_tidy.py` does).
2. **Off-menu responses in Experiment 1.** 137 of 6,720 condition-2 trials (2.04 %) and 44
   of 6,720 condition-3 trials (0.65 %) record a `response` that is neither `correctColor`
   nor `wrongColor` — subjects pressed a key for an unavailable colour. These trials are
   scored `correct = 0` and are silently absorbed into the "wrong" bin of
   `respPattern_cond2`, so the aggregate matrix's `n_correct + n_wrong` includes them.
   Decide explicitly whether to drop them. Experiment 2 has zero off-menu responses in
   15,000 two-alternative trials.
3. **`respPattern_*` matrices are stored as `uint8` in Exps 1 and 2.** Observed maxima are
   58 (Exp 1) and 235 (Exp 2), so no saturation occurred, but the headroom is thin — do not
   re-derive with more trials into the same dtype.
4. **Experiment 1 condition 3 (advance warning) is unanalysed in the paper.** The Methods
   say: *"For the purposes of the current analyses, we only analyzed the four- and
   two-alternative conditions. The advanced warning condition and the confidence ratings
   were not analyzed."* The data are fully present (210 trials/subject with menu identity),
   and mean accuracy is 0.8504 vs 0.7802 for the after-the-fact menu.
5. **No trial-level stimulus record in Experiment 1.** Exps 2 and 3 store the exact 7×7
   grid (`stimOrders`); Exp 1 stores only the dominant colour. Exp 4 stores the three
   directions in radians (`direction_used`) but not the dot assignments.
6. **Exp 4 `proportion_used` varies by subject** (0.50, 0.55 or 0.60 dominant-dot
   proportion; individually thresholded), so Exp 4 subjects are not on a common stimulus
   scale.
7. Subject numbering is independent per experiment; each subject took part in exactly one
   experiment (63 total).
8. Exp 3's `response(:,2) == 0` / `correct(:,2) == 99` are the sentinels for "no second
   answer requested" (second answers were solicited on ~40 % of error trials only).
9. The OSF bulk-zip endpoint answers `501` to `HEAD`; it serves 200 on `GET`.

## The authors' model, in their words

An **independent Gaussian, unit-variance, means-only** latent-activation model, fitted to
the full-menu response distribution and then used to predict restricted-menu choices with
**zero free parameters**.

Setting it up (Methods, "Model development for Experiments 1–3"):

> "We assumed that each of the four types of stimuli (red, blue, green, or white being the
> dominant color) produced variable across-trial activity corresponding to each of the four
> colors. We modeled this activity as Gaussian distributions whose mean (μ) is a free
> parameter and variance is set to one. However, in our experiments, the perceptual
> decisions only depended on the relative values of the activity levels and not on their
> absolute values. In other words, adding a constant to all four μs for a given dominant
> stimulus would result in equivalent decisions. Therefore, without loss of generality, we
> set the mean for the activity corresponding to each dominant color as 0. This procedure
> resulted in 12 different free parameters such that for each of the 4 possible dominant
> colors there were 3 μs corresponding to each of the nondominant colors. Finally, we
> included an additional parameter that models the lapse rate."

Direction of the lapse-rate bias, in their own words:

> "Note that the inclusion of a lapse rate parameter has a greater influence on percent
> correct in the two-alternative compared to the four-alternative condition because overall
> performance is higher in the two-alternative condition. Therefore, introducing a lapse
> rate parameter favors the population model by leading to predictions of lower performance
> in the two-alternative condition (which helps the population model since it consistently
> predicts higher performance than what was empirically observed)."

Fitting to the full menu, then predicting the restricted menu:

> "In order to compare the population and summary models, we first had to develop a model
> of the sensory representation. We created this model using the four- and six-alternative
> conditions in Experiments 1 and 2, and the first answer in Experiment 3. The population
> and summary models were then used to make predictions about the two-alternative condition
> in Experiments 1 and 2, and the second answer in Experiment 3. **These predictions were
> made without the use of any extra parameters.**"

How the likelihood is computed (Methods, "Model fitting and model comparison"):

> "For all four experiments, we fit the models to the data as previously using a maximum
> likelihood estimation approach. The models were fit to the full distribution of
> probabilities of each response type contingent on each stimulus type:
> Log likelihood = Σ_{i,j} log(p_ij) × n_ij, where p_ij is the predicted probability of
> giving a response i when stimulus j is presented, whereas n_ij is the observed number of
> trials where a response i was given when stimulus j was presented. … Because the
> analytical expressions to obtain p_ij are difficult to compute, we derived the model
> behavior for every set of parameters by numerically simulating 100,000 individual trials
> with that parameter set. Model fitting was done by finding the maximum-likelihood
> parameter values using simulated annealing. Fitting was conducted separately for each
> subject."

Experiment 4 (Methods, "Model development for Experiment 4"):

> "we fitted a model of the sensory representation to the three-alternative condition …
> Unlike Experiments 1–3 where the categories of stimuli were fixed, here the dominant
> direction of motion was chosen randomly (from 0° to 360°) on every trial. Therefore, the
> model only had parameters for the heights of the nondominant and dominant directions of
> motion. Because, just as in the previous experiments, adding a constant to all both
> parameters would result in identical decisions, the parameters for the nondominant
> direction was fixed thus leaving us with a single free parameter. Once the model was fit
> to the data from the three-alternative condition, the population and summary models had
> no free parameters when applied to the data from the two-alternative condition."

The code confirms it is a Thurstone Case-V / probit structure. In
`Experiment 1/codes/analysis/fitting_extended/logL_func_extended.m` the activations are
drawn as `normrnd(repmat(mu_relevant,1,...), ones(...))` and the full-menu response is
`[~, response_cond1] = max(relevantSignal)`. In `step3_test_fitting.m` the population
model's restricted-menu prediction is exactly a pairwise Gaussian comparison:

```matlab
p = sum(signal(stimPresented,stimPresented,:) >= signal(stimPresented,stimPair,:)) / N;
```

whereas the summary model conditions on the argmax:

```matlab
numCorrect = sum(response_cond1{stimPresented}==stimPresented);
numWrong   = sum(response_cond1{stimPresented}==stimPair);
```

Supplementary Methods gives the closed form for the accuracy gap. Their derivation, and
the fact that it is signed:

> "From here we obtain that the difference between the accuracy of the population and
> summary models in the 2-alternative condition is … Because the dominant stimulus is at
> least as likely to produce the second highest than the 4th highest activation, …, which
> means that the population model predicts higher accuracy in the 2-alternative condition
> compared to the summary model."

## What they conclude about the shortfall

The Gaussian population model — the one whose restricted-menu prediction is the
Thurstonian nesting prediction — **overpredicts** two-alternative accuracy in every
experiment. Verbatim:

**Experiment 1:**

> "Indeed, based on the performance in the four-alternative condition (average accuracy =
> 69.2%, chance level = 25%), the population and summary models predicted an average
> accuracy of 84.2% and 79.7% in the two-alternative condition, respectively. Compared to
> the actual subject performance (average accuracy = 78%), the population model
> overestimated the accuracy in the two-alternative conditions for 29 of the 32 subjects
> (average difference = 6.21%; t(31) = 8.19, p = 3.02 × 10⁻⁹, 95% CI = [4.7%, 7.8%]).
> Surprisingly, the summary model also overestimated the accuracy in the two-alternative
> condition but the misprediction was much smaller (average difference = 1.72%; t(31) =
> 2.35, p = 0.025, 95% CI = [0.2%, 3.2%])."

**Experiment 2:**

> "We found that the average accuracy in the two-alternative condition (71.6%) was slightly
> underestimated by the summary model (predicted accuracy = 70.1%; t(9) = −2.76, p = 0.022,
> 95% CI: [−2.7%, −0.3%]) but was again significantly overestimated by the population model
> (predicted accuracy = 77.5%; t(9) = 9.41, p = 5.92 × 10⁻⁶, 95% CI [4.5%, 7.3%])."

**Experiment 3** (second answer rather than a two-alternative menu):

> "We found that task accuracy for the second answers was 29.6%. This value was greatly
> overestimated by the population model, which predicted accuracy of 40.9% (t(9) = 7.04,
> p = 6.09 × 10⁻⁵, 95% CI = [7.7%, 15%])."

**Experiment 4:**

> "As in the previous experiments, we observed that the population model consistently
> overestimated the accuracy in the two-alternative condition (observed accuracy = 83.7%;
> predicted accuracy = 85.9%, t(10) = 4.31, p = 0.002, 95% CI [1.1%, 3.3%]), whereas the
> summary model predicted the observed accuracy well (predicted accuracy = 83.1%,
> t(10) = −1.37, p = 0.2, 95% CI = [−1.7%, 0.4%])."

**Overall:**

> "The results across all experiments showed that the population model that assumes no loss
> of information from sensory to decision-making circuits did not provide a good fit to the
> data. Instead, the summary model, which assumes that decision-making circuits represent a
> reduced form of the sensory distribution, consistently provided a substantially better
> fit. These results strongly suggest that deliberate decision making for multiple
> alternatives only has access to a summary form of the sensory representation."

Note that the shortfall is not one-signed for the *summary* model, and they say so:

> "This point is further underscored by the fact that when predicting the accuracy in the
> two-alternative condition, the summary model showed a slight but systematic overprediction
> in Experiment 1 but underprediction in Experiment 2 (though it was better calibrated in
> Experiment 4)."

Model-comparison magnitudes (fits to the whole restricted-menu response distribution, not
just accuracy; both models have zero free parameters at the prediction stage, so
AIC = −2·log L and AIC = BIC = AICc):

| Exp | mean ΔAIC favouring summary | subjects favouring summary |
|---|---|---|
| 1 | 24.30 (total 777.63) | 30 / 32 |
| 2 | 57.79 (total 577.94) | 9 / 10 |
| 3 | 18.05 vs Summary&Random; 37.29 vs Summary&Strategic (totals 180.46 / 372.93) | — |
| 4 | 5.47 (total 60.15) | 9 / 11 |

Intermediate models (2-Highest, 3-Highest, 2-Attention, 3-Attention) were also tested and
all lost to the summary model; per-subject numbers are in the Supplementary Notes
(`MOESM1`) and the `Sup. Figure 1/2/6` sheets of the Source Data.
