# Comay et al. (2023) — "dud alternative" confidence experiments

Trial-level data for five experiments in which participants choose the largest of
two, three (or up to five) alternatives, where the extra alternatives are
deliberately much smaller / farther and are almost never correct ("duds").

## Citation

Comay, N. A., Della Bella, G., Lamberti, P., Sigman, M., Solovey, G., &
Barttfeld, P. (2023). The presence of irrelevant alternatives paradoxically
increases confidence in perceptual decisions. *Cognition*, **234**, 105377.
https://doi.org/10.1016/j.cognition.2023.105377

Preprint: Comay et al. (2021), PsyArXiv, https://doi.org/10.31234/osf.io/zq7gw
(landing page: https://osf.io/preprints/psyarxiv/zq7gw)

Corresponding author: Nicolás A. Comay, nicocomay@gmail.com

## Repository

Public GitHub repository, no license file, declared in the PsyArXiv record's
`data_links` field (`has_data_links: "available"`):

- https://github.com/nicolascomay/confidence_dud

Downloaded 2026-08-17 from the default branch `main`.
Upstream HEAD at download: `4d2370b82241c89efa8e7d823c94c11da48adfad`
("Add files via upload", 2024-01-30T06:29:43Z).

Direct download URLs (all verified HTTP 200):

- https://raw.githubusercontent.com/nicolascomay/confidence_dud/main/data/data_experiment1.csv
- https://raw.githubusercontent.com/nicolascomay/confidence_dud/main/data/data_experiment2.csv
- https://raw.githubusercontent.com/nicolascomay/confidence_dud/main/data/data_experiment3.csv
- https://raw.githubusercontent.com/nicolascomay/confidence_dud/main/data/data_experiment4.csv
- https://raw.githubusercontent.com/nicolascomay/confidence_dud/main/data/data_experiment5.csv

## Local layout

The upstream tree was flattened (the clone's `.git` was discarded; no nested git
repository is committed here).

```
duds/
  SOURCE.md            <- this file
  README_upstream.md   <- verbatim upstream README.md (authors' codebook)
  data/                <- the five trial-level CSVs
  scripts/             <- upstream R analysis + JATOS/JS experiment code
                          (experiment1..5/, modeling/)
```

`scripts/modeling/data_experiment1.csv` is a byte-identical duplicate of
`data/data_experiment1.csv`.

## Files

| File | Bytes | Data rows | Cols | Subjects | SHA-256 (first 16) |
|---|---|---|---|---|---|
| `data/data_experiment1.csv` | 2,138,548 | 11,751 | 24 | 99 | `53018b88f97f53d2` |
| `data/data_experiment2.csv` | 1,726,875 |  8,368 | 26 | 18 | `839d484addd013c9` |
| `data/data_experiment3.csv` |   982,150 |  6,124 | 28 | 52 | `ba304c4666eb2909` |
| `data/data_experiment4.csv` |   646,341 |  3,879 | 25 | 33 | `3e9edc16c1624ed1` |
| `data/data_experiment5.csv` |   587,493 |  3,680 | 27 | 31 | `8ceede4bd9e65436` |

Row counts exclude the header. One row = one trial.

### Format caveats (important)

- **Experiments 1, 2, 3** are `;`-delimited with `,` as the decimal mark and a
  leading unnamed index column. **Experiments 4, 5** are `,`-delimited with `.`
  decimals and no index column. Experiment 5 additionally quotes some numeric
  fields.
- Several float columns in experiments 1–3 (and `Angle`, `xtarget`, `ytarget` in
  exp 5) were written with **spurious thousands separators**, e.g.
  `589,164,114,763,711` for `5.89164114763711` and `8,636,472,363,424` for
  `8636.472363424`. These are corrupted on export and must be repaired by
  stripping separators and re-inferring the decimal point, or ignored.
  **Affected: `Angle`, `Step_Angle`, `Area1`, `Area2`, `Area3`, `Confidence`
  (exp 1), `xtarget`, `ytarget`.**
- The columns needed for a restriction/menu test are **integers or booleans and
  are unaffected**: `Nalternativas`, `Response`, `Correct`, `binary_correct`,
  `BiggerCircleOrSquare`, `PosBig`, `n3/n4/n5SquareOrCircle`, `color_nube1..3`,
  `color_buttons1..3`, `correct_color`, `correct_cloud`, `Nsujeto`,
  `Trial_number`. `StimVal`, `StimVal3`, `StimValDud` and `distance_ratio` are
  one-decimal values and parse cleanly with the right decimal mark.

## Column names

**Experiment 1** (`;`, 24 cols; first column unnamed row index):
`, Code, Gender, Age, Mobile, Trial_number, Angle, Step_Angle, Area1, Area2,
Area3, n3SquareOrCircle, BiggerCircleOrSquare, PosBig, StimVal, StimVal3,
Nalternativas, RT_type1, Response, Correct, Confidence, RT_Confidence, Nsujeto,
binary_correct`

**Experiment 2** (`;`, 26 cols; two unnamed columns):
same as exp 1 plus `n4SquareOrCircle, n5SquareOrCircle`, and `StimVal3` is named
`StimValDud`.

**Experiment 4** (`,`, 25 cols):
`Code, Gender, Age, Mobile, Trial_number, Angle, Step_Angle, Area1, Area2,
Area3, n3SquareOrCircle, BiggerCircleOrSquare, PosBig, StimVal, StimVal3,
Nalternativas, RT_type1, Response, Correct, Confidence, RT_Confidence, Nsujeto,
binary_correct, squareResp, circleResp`

**Experiment 3** (`;`, 28 cols) / **Experiment 5** (`,`, 27 cols):
`[index,] Code, Gender, Age, Mobile, Trial_number, Angle, distance_ratio,
xtarget, ytarget, sd, nsamples, color_nube1, color_nube2, color_nube3,
color_buttons1, color_buttons2, color_buttons3, correct_color, correct_cloud,
Nalternativas, RT_type1, Response, Correct, Confidence, RT_Confidence, Nsujeto,
binary_correct`

## Column meanings

Authors' codebook is reproduced verbatim in `README_upstream.md`. Meanings
below are the codebook plus what was verified against the experiment JavaScript
in `scripts/*/experiment_code/js/` and against the data itself.

### Size task (experiments 1, 2, 4) — "which geometric figure is largest?"

| Column | Meaning |
|---|---|
| `Nalternativas` | Number of figures shown on the trial. Exp 1 & 4: 2 or 3. Exp 2: 2, 3, 4 or 5. |
| `Response` | **Which figure was chosen.** `1` = the circle (canvas 1), `2` = the square (canvas 2), `3` = the first dud (canvas 3), `4`/`5` = further duds (exp 2 only). Verified in `create_stim.js`: the circle is always drawn in `context1`, the square always in `context2`, the distractor always in `context3`. |
| `BiggerCircleOrSquare`, `PosBig` | `1` = the circle is the largest figure, `2` = the square is. (Same variable twice; `PosBig` was for debugging.) Verified: `Correct == (Response == BiggerCircleOrSquare)` on **all** 11,751 / 8,368 / 3,879 rows of exp 1 / 2 / 4. |
| `n3SquareOrCircle` | Shape of the dud: `1` = square, `2` = circle, `0` = no dud present. Note the dud's *shape* can duplicate one of the two main figures; its *identity as an alternative* is still `Response == 3`. |
| `n4SquareOrCircle`, `n5SquareOrCircle` | Same for the 4th and 5th alternatives (exp 2). |
| `StimVal` | Area of the second-largest main figure as a proportion of the largest: 0.7, 0.8, 0.9, 0.93, 0.95. This is the difficulty manipulation. |
| `StimVal3` / `StimValDud` | Area of the dud as a proportion of the second-largest figure: 0.1–0.6 (`0` when no dud). |
| `Area1, Area2, Area3` | Absolute pixel areas of the largest, second, and dud figures. `Area2 = Area1 * StimVal`, `Area3 = Area2 * StimVal3`. **Corrupted number formatting in exp 1 & 2 — see caveats.** |
| `Correct` | `TRUE`/`FALSE`; `binary_correct` is the 1/0 version. |
| `Confidence` | Reported confidence, 0–1 in the files (README says 0–100). |
| `RT_type1`, `RT_Confidence` | Decision and confidence-report RT in ms. |
| `Angle`, `Step_Angle` | Random rotation of the stimulus array; `Step_Angle` is ±2π/3 rad. |
| `Nsujeto`, `Code` | Subject index, and MD5 subject identifier. |
| `Mobile` | 1 = phone, 0 = computer. |
| `squareResp`, `circleResp` | Exp 4 only; one-hot indicators of `Response`. |

### Categorical task (experiments 3, 5) — "which coloured dot cloud is the target nearest?"

Three Gaussian dot clouds; clouds 1 and 2 sit adjacent (radius `R`, 120° apart),
cloud 3 sits far away at radius `2.5R` and is **never** the correct answer, i.e.
it is the dud. The whole array is rotated by a random `Angle`. Colours are a
random permutation of {red, green, blue} assigned to clouds 1, 2, 3 each trial
(`shuffle([0,1,2])` in `create_stim.js`); the three response buttons are always
in fixed screen order red, green, blue.

| Column | Meaning |
|---|---|
| `Nalternativas` | `2` or `3` — number of clouds shown *and* number of response buttons visible. `show_stim.js` explicitly **hides the button whose colour belongs to cloud 3** on 2-alternative trials. |
| `color_nube1`, `color_nube2`, `color_nube3` | Colour index of clouds 1, 2, 3: `0` = red, `1` = green, `2` = blue. Always a permutation of (0,1,2), including on 2-alternative trials, where `color_nube3` is the **excluded** colour. |
| `color_buttons1..3` | Always `0, 1, 2` — fixed button colour order. |
| `correct_color` | Colour index of the correct cloud. |
| `correct_cloud` | `1` or `2` only (cloud 3 is never correct). |
| `Response` | **Colour index chosen** (`0`/`1`/`2`), which maps back to a cloud via `color_nube1..3`. |
| `distance_ratio` | Difficulty: target's position between clouds 1 and 2, values 3, 2.5, 2.1 (2 would be exactly midway). |
| `xtarget`, `ytarget` | Target pixel coordinates. Corrupted formatting in places — see caveats. |
| `sd`, `nsamples` | Cloud SD (60) and dots per cloud (375), responsive to screen size. |
| `Correct`, `binary_correct`, `Confidence`, `RT_type1`, `RT_Confidence`, `Nsujeto`, `Code`, `Mobile`, `Gender`, `Age` | As above. Verified: `Correct == (Response == correct_color)` on all 6,124 / 3,680 rows. |

## Example rows

`data/data_experiment4.csv` (header + first two data rows):

```
"Code","Gender","Age","Mobile","Trial_number","Angle","Step_Angle","Area1","Area2","Area3","n3SquareOrCircle","BiggerCircleOrSquare","PosBig","StimVal","StimVal3","Nalternativas","RT_type1","Response","Correct","Confidence","RT_Confidence","Nsujeto","binary_correct","squareResp","circleResp"
"41a7b3f9312c7ebd5aaf7a21adda70ac","f",22,0,2,1.20540538246632,-2.0943951023932,9601.76202784825,7681.4096222786,4608.84577336716,1,2,2,0.8,0.6,3,2581,2,TRUE,0.921364985163205,3058,1,1,1,0
"41a7b3f9312c7ebd5aaf7a21adda70ac","f",22,0,3,2.87636111772903,-2.0943951023932,5430.52890212488,3801.37023148742,1520.54809259497,0,1,1,0.7,0,2,1757,1,TRUE,0.961424332344214,2863,1,1,0,1
```

Read the first row as: 3 alternatives on screen (circle, square, and a square
dud); the square was the largest; second figure was 0.8 of the largest; the dud
was 0.6 of the second; the subject chose the square (`Response = 2`) and was
correct.

`data/data_experiment3.csv` (header + first data row, `;`-delimited):

```
;Code;;Gender;Age;Mobile;Trial_number;Angle;distance_ratio;xtarget;ytarget;sd;nsamples;color_nube1;color_nube2;color_nube3;color_buttons1;color_buttons2;color_buttons3;correct_color;correct_cloud;Nalternativas;RT_type1;Response;Correct;Confidence;RT_Confidence;Nsujeto;binary_correct
1;edd6dd0b1f1d647652335f32a3aada1b;;f;25;0;1;601,396,094,148,694;3;683;256,666,666,666,667;60;375;2;1;0;0;1;2;1;2;2;6276;1;TRUE;1;1345;1;1
```

Read as: 2 clouds shown, cloud 1 = blue and cloud 2 = green, so **the menu was
{blue, green} and red was unavailable**; the correct cloud was cloud 2 (green);
the subject responded green (`Response = 1`) and was correct.

## Does the data record which alternatives were available and which was chosen?

**Yes, for all five experiments.** Both halves of the restriction test are
present at trial level.

**Experiments 1, 2, 4 (size task).** The menu is
`{circle, square}` when `Nalternativas == 2` and
`{circle, square, dud(s)}` when `Nalternativas >= 3`; `Response`
identifies the chosen element on the same 1/2/3(/4/5) scale, verified against
the drawing code. The alternatives are size-defined roles rather than fixed
labelled goods, so a restriction test must condition on `StimVal` (and
`BiggerCircleOrSquare` to fix which shape is the large one). Doing so works:
e.g. exp 1, 2-alternative trials, circle largest, `StimVal = 0.95` gives
`p(choose circle) = 0.634` over n = 588; the matching 3-alternative cells are
directly comparable.

**Experiments 3, 5 (categorical task).** Cleanest case: the available set is an
explicitly recorded subset of three *labelled* response buttons.
`Nalternativas == 2` gives menu `{color_nube1, color_nube2}` with `color_nube3`
excluded; `Nalternativas == 3` gives all three. All three pairs occur, roughly
balanced (exp 3: red+green 976, red+blue 1045, green+blue 1041 trials).
Confirmed empirically: across **all** 3,062 (exp 3) and 1,838 (exp 5)
two-alternative trials, `Response` never equals `color_nube3` — the excluded
button was genuinely unavailable.

### Caveats for the restriction test

1. **The dud is essentially never chosen.** Exp 1: 7 dud choices out of 5,877
   three-alternative trials (0.12%). Exp 2: 3 / 2,100 and 4 / 2,084. Exp 4:
   0 / 1,945. Exp 3: 4 / 3,062. Exp 5: 7 / 1,842. So the interesting quantity is
   whether the *relative* split between the two real alternatives shifts when the
   dud is added, not the dud's own share. Aggregate marginals are close: exp 1
   circle share 0.464 (2-alt) vs 0.461 (3-alt); exp 4 0.482 vs 0.474. Any
   regularity/restriction violation will have to be found conditional on
   difficulty and/or per subject, not in the pooled marginals.
2. **Colour labels in exp 3/5 are exogenous random labels.** The colour
   permutation is redrawn every trial, so "red" carries no intrinsic value and
   the six-pair structure is not a fixed-item-set structure in the economic
   sense. The behaviourally meaningful item labels are the cloud roles
   (cloud 1 / cloud 2 near the target, cloud 3 far = dud), which reduces exp 3/5
   to the same two-real-alternatives-plus-dud design as exp 1/2/4. Pooled
   choice shares within each colour pair are near 50/50 as expected
   (e.g. exp 3 red+green: 456 vs 520).
3. **Online, unsupervised sample.** Recruited via web; `Mobile` flags phone
   participants. Subject counts differ a lot across experiments (99, 18, 52, 33,
   31), and exp 2 has many trials per subject (up to 480) while exp 1 has ~120.
4. **Trial numbering starts at 2 in exps 4 and 5** (first trial appears to be
   dropped as practice); exps 1–3 start at 1, and exp 1 skips index 2 for the
   first subject, so trial indices are not gap-free.
5. **No license** is declared on the upstream repository. Cite the paper and the
   repository; check with the corresponding author before redistribution.
6. Experiment 3 is the experiment for which the authors report the confidence
   effect **did not** replicate; experiments 4 and 5 vary stimulus presentation
   time. None of this affects the availability of the categorical response.
