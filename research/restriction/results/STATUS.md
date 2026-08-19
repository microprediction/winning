# Referee response: state as of 2026-08-14

## Done

Three substantive complaints, all acted on.

1. **Wrong null in the held-out test.** `luce_null.py` generates complete rankings
   from an exact Plackett-Luce process whose worths are the observed first-place
   shares, then reruns the identical five-fold pipeline. Eleven of twelve datasets
   exceed the null at p <= 0.005. Sports participation (n=130) does not: excess
   -0.0000, p = 0.51. Its gain is finite-sample shrinkage and it is now excluded
   from the paper's claims. Wherever n >~ 2000 the null is *negative*, so the old
   comparison with zero understated the effect. The properness sentence is deleted.

2. **Tail-thickness claim retracted.** Replaced by a proposition on the curvature of
   the reverse hazard r = f/F: log r concave => removal contracts the favourite's
   log odds; log r affine => equality, which is the Gumbel and recovers Yellott.
   Gaussian has (log r)'' = -Var(Z|Z<x) < 0. Verified in `hazard_contraction.py`.
   The referee's counterexample reproduces: on shares (.88,.07,.05) the Gaussian
   gives 0.0384 and t(3) gives 0.0555, so the ordering reverses with configuration.

3. **Saturated benchmark is not a ceiling.** E[L_sat] >= L_oracle, denominator too
   small, fraction biased upward. Retitled and restated. Median share corrected
   from 43% to 39% (the script took the upper of two middle values).

Also: loaders unified so all tables cover the same twelve collections (Netflix,
dots, puzzles recovered as separate rows); contraction table regenerated with
respondent-bootstrap intervals that recompute shares, calibration and both slopes;
occupational prestige restored under a shared add-alpha convention and it posts the
largest gain; all inputs committed under `research/restriction/data` at 1.9MB so nothing
reads from a temp directory; winner-orientation clash between Equation (2), the
pairwise formula, the figure and the lattice code resolved onto highest-wins.

**Ordered-outcome paragraph cut.** Pricing an exacta by removing the winner and
re-running is a sequential heuristic, not the race's ordering law. Measured against
the calibrated race by simulation it misprices cells by up to 3.4x, which exceeds
the effect it was meant to demonstrate.

## Running (launched 2026-08-14, output in this directory)

- `luce_null_200.txt` -- full 12 datasets x 2 alphas x 200 replicates. Hours. The
  only cell the paper currently takes from a shorter run is Sushi at alpha=0.5.
(`gain_by_size.txt` finished and is now Table 3 in the paper.)

## Landed after the status note was first written

Gain by menu size (`gain_by_size.txt`, now a table in the paper). The pairwise gain
is two to four times the all-subsets aggregate (Sushi +0.0412 against +0.0111; GSS
socialization +0.0130 against +0.0055), and the gain decays monotonically to exactly
zero at |T| = K, where nothing has been removed and the two accounts must coincide.
That last column is a pipeline check. So the aggregate is a conservative estimand,
and it is kept as the headline because it is fixed in advance rather than selected
after the fact.

## Not done

- Per-dataset flow table: source, K, raw and retained counts, tie handling, whether
  subsets are observed or ranking-induced, pooling, inclusion status, seed.
- The held-out bootstrap resamples respondent losses with the fitted training models
  held fixed, so it omits calibration uncertainty. The lambda-table bootstrap does
  recompute everything. This asymmetry should be stated or fixed.
- Consumer-products experiment is still not a primary result, though it is the only
  dataset observing real subset choice.
- Exacta and trifecta pricing redone against the exact ordering law.
- Nothing is pushed. The referee could not see `research/restriction` on main for that
  reason. A tagged commit or archival DOI is needed before circulation.

## Counter-evidence to find room for (2026-08-17)

Meyer-Grant, Kellen, Harding & Singmann, "Extreme-Value Signal Detection Theory for
Recognition Memory", OSF preprint qhrfj, project gtzu7, submitted Dec 2025, not yet
published. This is the sharpest challenge to the paper and must be cited.

They prove Gumbel-min uniquely predicts accuracy invariance as the choice set grows
uniformly, where Gaussian predicts change, and they find the invariance: chi2(3)=0.69,
p=.876, BF=679 for the null, in 253 participants. Their predictive benchmarking summary:
"Across all predictive tests, a pattern was clear -- the Gumbel_min model outperformed the
Gaussian."

The property they validate is Yellott's own condition, invariance under uniform expansions
of the choice set, which Yellott showed equivalent to the axiom. So they independently
confirm the axiom in recognition memory, the same domain where our Utochkin analysis has
the Gaussian race beating renormalization on nested foils.

Both can hold. Their test grows the set with fresh items and measures accuracy; ours removes
named foils and measures redistribution. Their own footnote suggests the reconciliation:
the invariance "breaks down when systematic similarity among stimuli is introduced because
latent strengths cease to be independent", which is exactly Utochkin's same-category foil.
Our similarity split found contraction equal across same- and cross-category foils, which is
in tension with that and worth re-examining.

Not usable as data: both Kellen programs draw fresh items per trial and per set size. Zero
word reuse across 110, 103 and 359 participants, verified directly from full_list.

Also: their Appendix B swaps the 252 and 253 participant counts relative to the posted CSVs.

## The jury data: first clear loss for both maps (2026-08-17)

Ormston, Chalmers, Leverick, Munro and Murray (2019), Scottish Jury Research, Scottish
Government, ISBN 978-1-83960-194-1. 863 deliberating jurors, 64 juries, fully crossed
2x2x2x2. Strict nesting: 32 juries had {Guilty, Not guilty}, 32 had {Guilty, Not guilty,
Not proven}. Trial films were "entirely identical (within trial type), with the exception of
the very final section, in which the judge tells the jury about the verdicts available".

Pre-deliberation shares: three verdicts G 28, NG 23, NP 50; two verdicts G 38, NG 62.
Renormalizing the three-verdict shares onto the survivors predicts G at 28/51 = 54.9 against
an observed 38, an error of 17 points. Post-deliberation: predicted 61.1 against observed 31,
an error of 30 points.

Both defaults fail, because the ordering REVERSES. Guilty leads Not guilty 28 to 23 with
three options and trails 38 to 62 with two. Contraction moves odds toward even and never
crosses over, so the Gaussian race cannot fix this either; it fails slightly less.

The mechanism is what independence forbids: Not proven is a near-substitute for Not guilty,
so deleting it returns its mass to Not guilty rather than proportionally. Debreu's blue-bus
objection in a courtroom, on identical stimulus films.

This belongs in the paper as the boundary condition. It says the Gaussian default is for
menus whose alternatives are not near-substitutes, and it is the strongest such statement
available because the stimulus is fixed by construction.

Corroborating and free: Curley et al. (2022), Psychiatry Psychology and Law 29(3):323-344,
eprints.gla.ac.uk/236471/1/236471.pdf, N=128, two verdicts G 42 NG 86, three verdicts
G 26 NG 22 NP 80, same direction. Within-subject so rows are not independent.

To obtain by library request: Vidmar (1972) JPSP 22(2):211-218 plus Larntz (1975) JPSP
31(1):123-125. If Vidmar really ran all seven non-empty subsets of the three homicide
charges with acquittal always available, that is a complete power-set design on one case and
the best Block-Marschak material in this entire search. Unverified from the primary source.

## Prior art that narrows the novelty claim (2026-08-17)

The CONSTANT-RATIO RULE is Luce's renormalization under another name, and it has been tested
against nested response sets since the 1950s. This literature was invisible to every search
term used until now and it materially narrows what this paper can claim as new.

Townsend & Landon (1982), J Math Psychol 25(2):119-162, doi 10.1016/0022-2496(82)90009-8.
Ran this paper's experiment: master set {A,E,F,H,X} with nested subsets {A,E,X}, {F,H,X},
{A,E,F,H}, blocked and counterbalanced, 4 subjects x 16 sessions, 240 presentations per
letter, and they PUBLISH FULL CONFUSION MATRICES PER SUBJECT in Tables 1-4. Their result is
ours and their diagnosis is our boundary condition: CRR fits {A,E,F,H} well and {F,H,X}
badly because confusions concentrate into the similar pairs A,E and F,H rather than
spreading proportionally. They cite Debreu (1960) by name. Free via Wayback:
web.archive.org/web/20160916190658if_/http://www.indiana.edu/~psymodel/papers/towlan82.pdf

Rouder (2004), Psych Review 111(1):80-93. States our result in the older vocabulary:
conditioning on a reduced choice set is "somewhat efficient", better than guessing
correction predicts but WORSE than Luce/CRR normalisation predicts. That is contraction,
reported in 2004. Re-analyses Townsend & Landon and Rouder (2001).

Morgan (1974), J Math Psychol 11:107-123, formalised a likelihood-ratio test of CRR and
rejected it on Clarke's and Egan's data. Hodge (1967) tested nested ensembles of 8, 4 and 2
and found CRR failures concentrated on similar pairs. Elliffe & Davison (2010) show a clean
2-vs-4 violation in pigeons.

WHAT THIS MEANS. "Renormalization fails under restriction" is not new and the paper must not
imply it is. What remains new: that the Gaussian race is the better PARAMETER-FREE DEFAULT,
scored out of sample against an explicit shrinkage null, across populations spanning
preference, recognition memory, perception, verdicts and clicks; and that the failures share
a structure (near-substitutes, quality-changing removal, concentrated shares).

Also usable as data, needing an author request: Rouder (2001) Psych Science 12(4):318-322,
2AFC {W,E} vs 6AFC {Q,W,E,R,T,Y} masked letters, footnote offers the frequency matrices.
Gummerman (1973), Bull Psychonomic Soc 2(6):365-367, free, reprints 1971 data with nested
subsets and finds NO small-set advantage after guessing correction, which is the opposite
direction from Waszak.

Waszak et al. (2009), Psych Research 73(1):114-122, the masked-symbol target: nesting is
literal, {#, thorn} inside {#, thorn, ampersand, ash}, n=226 ages 6-88, and the paper already
reports the nested comparison on the same two stimuli. No trial data, but Table 1 gives 132
mean d-prime values with SDs, analytically invertible to percent correct. Their anomaly runs
the wrong way: d-prime HIGHER with four alternatives than two, F(1,216)=58.84, which their
own footnote attributes to the added symbols being easier, so guessing redistributes
non-proportionally onto the retained pair.

## Second boundary condition, with two instances (2026-08-17)

McMurray, Aslin, Tanenhaus, Spivey & Subik (2008), JEP:HPP 34(6):1609-1631,
pmc.ncbi.nlm.nih.gov/articles/3011988/, free. Literal nesting twice over: {b,p} inside
{b,p,l,sh} on identical synthetic CV stimuli (Exp 2 vs 3) and on six word continua (Exp 5 and
McMurray 2002 vs Exp 4). Between subjects.

The reported result is a boundary case for us. The two added labels absorbed only 0.8 per cent
of responses, yet the b:p identification slope fell from 0.99 to 0.82, t(42)=4.3, p<.001. So
removing them makes the surviving discrimination MORE extreme. Renormalization predicts no
change, since the removed options carried almost no mass, and contraction predicts movement
toward even. Observed is the opposite of both.

Same direction as the Zoanetti exam data, where deleting a distractor raised the correct
answer share by more than proportional (+2.65pp, t=3.04, p=.004). Two domains, one mechanism:
removal made the task easier, so the surviving alternatives changed in discriminability.

THE DISCUSSION NOW HAS TWO BOUNDARY CONDITIONS, EACH WITH TWO INSTANCES.

1. Near-substitutes. Deleting an option returns its mass to its neighbour rather than
   proportionally, and the ordering can reverse. Scottish verdicts (Not proven and Not
   guilty); Townsend & Landon 1982 (confusions concentrate into A,E and F,H).

2. Quality-changing removal. The surviving alternatives become easier to discriminate, so
   they are not the same alternatives before and after. Exam distractors; phoneme labels.

Plus the pre-existing third: both defaults agree wherever shares are concentrated, so
near-deterministic populations cannot answer the question at all. Low-cost gambles frame,
third-party ballot mass, machine data.

Data status for McMurray: identification curves only, digitizable from Figures 5, 7 and 9;
filler responses were discarded as false alarms so no 4-way matrix exists; trial-level files
need an author request to bob-mcmurray@uiowa.edu.

Barclay (1972), P&P 11(4):269-273, is the original nested-response-set speech experiment,
within-subject on the same tape, b/d/g then b/g. Worth citing but the 3-category proportions
were never printed and cannot be recovered from the published tables.

## Tones: third loss, and the cleanest near-substitutes case (2026-08-17)

research/restriction/tones.py on the digitized Stewart, Brown and Chater matrices. Calibrate on the
N=10 row for a stimulus, restrict to the middle labels, predict the N=6 or N=8 row.

  narrow N10->N6   renorm 1.1176  race 1.1309  gain -0.0133
  narrow N10->N8   renorm 1.3569  race 1.3620  gain -0.0051
  wide   N10->N6   renorm 1.0053  race 1.0226  gain -0.0173
  wide   N10->N8   renorm 1.2640  race 1.2697  gain -0.0057

Renormalization wins all four. The mechanism is visible in the matrices: they are strongly
banded, so a tone is confused almost entirely with its immediate neighbours. The alternatives
lie on a one-dimensional frequency continuum, so adjacent labels are near-substitutes by
construction, and removing the outer labels returns their mass to neighbours rather than
proportionally. Same mechanism as the Scottish verdicts, in a psychophysical setting, and the
loss is largest at N=6 where more of the continuum has been cut away.

THE BOUNDARY CAN NOW BE STATED AS A RULE. The race wins where alternatives are distinct
unordered items: consumer goods, news slates, memory foils, sushi, jokes, ballots. It loses
where they sit on a perceptual continuum (tones), or where removal changes the task (exam
distractors, phoneme labels), or where one removed option is a near-substitute for a survivor
(verdicts).

This also resolves the tension with Meyer-Grant. Our Utochkin similarity split found
contraction equal across same-category and cross-category image foils, which looked like
evidence for independence; tones show similarity mattering greatly. Both hold, because two
photographs from one category are far less confusable than two tones 6 per cent apart in
frequency. It is the continuum that breaks independence, not category membership.

## Three more nested-response-set sources, verified end to end (2026-08-17)

1. ROUDER LAB "chunk" EXPERIMENTS -- raw trial data, and the best unexploited lead.
github.com/PerceptionCognitionLab/data0/tree/master/1dMemory/chunk. Unpublished lab data in a
public repo, no licence, no README; the design must be read off the .C sources, where
C2R/C2.C contains blocktypestims[4][7]={{0,1},{2,3,4},{5,6},{0,1,2,3,4,5,6}} and the logged
stimulus and response are GLOBAL indices, so labels are identical across menus. Columns
sub ch wd blk trl bt set stim resp RT.
  c0 cond A,D: 12 lines, subsets {0-3},{4-7},{8-11} and {0-7},{4-11}, 35 subjects, 880 trials
  c0 cond C: 12 distinct PAIRS from the 12-line master, 14 subjects -- the sharpest instrument,
    since the same subject gives a 12-alternative and a binary choice over identical labels
  C2R and c2: 7 lines, subsets {0,1},{2,3,4},{5,6}, 47 subjects, 782 trials
  C3: order-counterbalanced, 12 subjects
Order is fixed full-subsets-full except in C3, so practice is confounded unless the pre/post
full-set brackets or C3 are used.

2. TOWNSEND & LANDON (1982), J Math Psychol 25(2):119-162. Printed obtained proportions to 3dp
for master {A,E,F,H,X} and nested {A,E,F,H}, {A,E,X}, {F,H,X}, per subject, 4 subjects, 240
trials per letter per block, spoken letter names so labels identical. Multiply by 240 to
recover integer counts, which self-checks the extraction. Only route is Wayback:
web.archive.org/web/20160916190658if_/http://www.indiana.edu/~psymodel/papers/towlan82.pdf

3. GETTY, SWETS, SWETS & GREEN (1979), Percept Psychophys 26(1):1-19, publisher OA at
link.springer.com/content/pdf/10.3758/BF03199856.pdf. Table 6 gives full 8x8 confusion
FREQUENCIES per observer; Table 8 the "8 by 4" experiment, same 8 stimuli with only 4
allowable responses, signal sets {1,2,5,6}, {3,4,5,6}, {1,3,5,7}. All 8 stimuli were presented
in the 8x4, so the four noise rows are forced errors and the nested comparison lives in the
signal rows. About 10x fewer trials per cell than the 8x8.

INTEGRITY CAVEAT from the agent's own correction. It relayed some Tier 2 detail from a
delegated thread that never returned -- it mistook its own timeout for completion. Treat as
unverified: the internals of Lacouture, Li & Marley (1998) Experiment 2a, including the label
sets {1..10} inside {3..7} inside {5,6}, n=16, 600 trials and PC .58/.76/.91, which it has
never seen because tandfonline blocks automated fetches. That is the primary target for this
literature, so open the PDF by hand first. The three Tier 1 items above it verified itself
from source files and downloaded PDFs.

Also corrected: the UK Data Service item is Adelman (2016) alone, and manipulates exposure and
decision time rather than set size, so it is off point. And Stewart's raw zip is confirmed
lost, not merely hard to find -- a CDX sweep shows only the HTML listings and a source_code
directory were ever captured.

## Yeon and Rahnev run: the race wins, and the experiment carries its own control (2026-08-17)

`research/restriction/yeonrahnev.py`, full table in `results/yeonrahnev.txt`. This is the first
dataset in the project where the restricted menu is both observed and unanticipated: the
observer sees the display, it goes, and only then is told the answer is one of two named
alternatives. Calibrate on that observer's full-menu row for that dominant item, predict
the pair, score out of sample.

  exp 1, 4 colours, after     384 cells  renorm 0.4942  race 0.4795  gain +0.0147  [+0.0118, +0.0179]
  exp 1, same pairs, before   384 cells  renorm 0.3967  race 0.3928  gain +0.0038  [+0.0006, +0.0072]
  exp 2, 6 symbols, after     300 cells  renorm 0.6276  race 0.5999  gain +0.0278  [+0.0234, +0.0324]

The +0.0278 at K=6 is the second largest effect in the project after the MIND news slates,
and the largest in a perceptual task. The null here is stronger than the one used elsewhere:
it resamples the calibration row as well as the pair counts, so it charges the race for
calibration noise. Null medians are -0.0010, -0.0009, -0.0036; excesses +0.0157, +0.0048,
+0.0314; all tails at the 0.005 floor of 200 replicates.

THE CONTROL IS THE POINT. Experiment 1 ran the identical pairs a third way, ANNOUNCED IN
ADVANCE, so the observer can aim attention at the pair. The authors never analysed that arm
and never saved it into dataForModeling.mat; `build_tidy.py` now recovers it from the raw
files, and recounting condition 2 the same way reproduces respPattern_cond2 exactly, which
is the extraction's self-check. In that arm the gain nearly vanishes, +0.0038 against
+0.0147, because accuracy rises from 0.7802 to 0.8504 and renormalization becomes almost
exactly right (error -0.0040) while the race now under-predicts (-0.0187). So the race's
advantage is not a property of these pairs or these observers. It appears when and only when
alternatives are genuinely withdrawn from a percept formed without knowledge of the menu.
No other dataset in the project has a within-experiment control of this kind.

THE CAVEAT, WHICH MUST TRAVEL WITH THE RESULT. In both genuine restriction arms BOTH maps
over-predict the favourite. Renormalization is +0.0691 and the race +0.0545 at K=4;
+0.0964 and +0.0643 at K=6. Contraction has the right sign and covers 21 and 33 per cent of
the error. Neither parameter-free default is calibrated here and a fitted noise term would
beat both. That over-prediction is Yeon and Rahnev's own headline -- their fitted
"population" model, which is close to our race, also predicts too well -- so their result is
independent evidence that the correction runs toward contraction, and it bounds what the
paper may claim.

Consistent with the boundary rule: colours and symbols are distinct unordered items, and the
race wins. The prediction to test next is Experiment 4, three motion directions, where the
alternatives sit on a circular continuum. Two reasons to expect little there: the three
directions are always 120 degrees apart with a random rotation, so the items are exchangeable
and there is no near-substitute asymmetry to exploit; and the 2-option blocks are blocked, so
the menu is known in advance, which the Experiment 1 control shows is the arm where the
effect disappears. Note also SOURCE.md's warning that Experiment 4's `c2.wrong_answer` is
misaligned in dataForModeling.mat, so exp4 must be built from the trial files.

Experiment 3's second answers are the ordering-law material, not a restriction test:
observing the first winner conditions the sample, so it needs the exact Gaussian ordering law
rather than remove-and-rerun. Keep it for the paper that follows.

## Getty 1979 scored, and the boundary acquires a computable diagnostic (2026-08-17)

`research/restriction/getty.py`, data in `data/getty`, output in `results/getty.txt`. Three
observers, eight complex sounds, all eight responses allowed in one experiment and only four
in another, a different four in each of three conditions, labels being the stimulus numbers.
The stimulus set never changes; only the menu does. Nothing else in the corpus isolates the
response-set manipulation that cleanly.

  all rows                        72 cells  renorm 0.8144  race 0.7872  gain +0.0272  excess +0.0343
  signal rows, favourite survives 36 cells  renorm 0.4504  race 0.4393  gain +0.0111  [+0.0021, +0.0225]
  non-signal, favourite removed   36 cells  renorm 1.1785  race 1.1351  gain +0.0434  excess +0.0529
  condition 1, signals {1,2,5,6}  24 cells  gain +0.0453  excess +0.0674
  condition 2, signals {3,4,5,6}  24 cells  gain -0.0127  excess -0.0330  tail 1.000
  condition 3, signals {1,3,5,7}  24 cells  gain +0.0490  excess +0.0665

THE PREDICTION WAS MADE BEFORE THE RUN. The source note flagged condition 2 in advance --
"a contiguous middle block, largest observed weight change" -- and asked whether the race
would fit it worse. It is the only condition the race loses.

AND THE MECHANISM IS MEASURABLE FROM THE MASTER MATRIX ALONE. Of the errors a signal
stimulus makes on the full eight-way menu, the fraction landing on another signal of that
condition is 0.103 for condition 1, 0.790 for condition 2, 0.335 for condition 3. The only
condition the race loses is the one whose survivors are each other's confusions. This turns
the near-substitute boundary from a taxonomy into a statistic an analyst can compute before
seeing any restricted-menu data, which is exactly the diagnostic Townsend and Landon used in
1982 and which the paper's Scope section now recommends.

Two caveats. Three observers, so the bootstrap resamples cells rather than observers and
understates uncertainty; only the signal-row interval excludes zero. And the authors
themselves report that observers retuned dimension weights to maximise discriminability of
whichever subset they had to identify, with feedback given only on that subset, so this is
also an instance of quality-changing removal. Both boundary conditions are present at once,
which is an argument for treating the condition-2 loss as the informative result and the
overall win as the weaker one.

Transcription check: every row sum in both tables reproduces the printed total, with one
exception where the printed TOTAL is wrong and the cells are right (J.K., condition 3,
stimulus 1: cells sum to 33, printed total 23).

## Townsend & Landon into the paper, with one reading corrected (2026-08-18)

`notes/crr/RESULT_townsend1982.md` is now Section "The rule's own founding dataset". Pooled
+0.0042 [+0.0022, +0.0064], excess +0.0053, p = 0.005, on 38 rows and 9,120 held-out trials,
reproduced from `townsend_rows.py`.

CORRECTION to the result note's framing. It reads the subset ordering as "removing the odd
letter out does essentially nothing, removing a near-substitute pair is where the race earns
its keep". That comparison is confounded: {A,E,F,H} withdraws ONE letter while {A,E,X} and
{F,H,X} withdraw TWO, and gain_by_size already shows gain rising with how much is removed and
going to exactly zero when nothing is. The unconfounded pair is {A,E,X} at +0.0040 against
{F,H,X} at +0.0093, and those two designs are structurally symmetric -- each removes a similar
pair and retains the other -- so the difference is not a similarity story. The paper says this
and claims no similarity effect from these subsets.

The gain-by-size table is now in the paper as well, since that argument needs it.

CITATION CORRECTION. The note `absolute_id/lee_1970.md` gives Lee as 1970, Perception &
Psychophysics 7(4). The DOI it records, 10.3758/BF03206305, resolves at Crossref to 4(4):
217-219, 1968. The paper cites Lee (1968). A correction is appended to the note.

## Caught up to main, and the empirical tree moved (2026-08-18)

origin/main had advanced 45 commits since the merge base of 2026-07-10, carrying a package
overhaul done on another machine. Merged into `machine-preference-paradox` at 58070ae. The
merge was textually clean and nothing from this side was lost except `pyproject.toml`, which
main supersedes with `setup.py`.

VERIFIED, because a merge that touches the package could in principle move a number: the
analysis pipeline imports numpy and `../polysemy_pilot/exact_analyze.py` and never the
`winning` package, and `tones.py` and `getty.py` reproduce their committed output byte for
byte, before and after the move. A safety tag `pre-main-merge-20260818` sits at the
pre-merge commit fab61d8.

`research/human` is now `research/restriction`, a sibling rename chosen so that the
`../polysemy_pilot` hop in all 27 scripts keeps working without edits. The seven text files
naming the old path were rewritten, including both manuscripts. A README now states the
question, the layout and the three standing conventions.

OPEN COLLISION, deliberately not resolved from this side. `src/winning` on this branch and
`winning/` from main are two different renovations of the same package. This side has the
ratings layer -- elo, glicko2, ratingsystem, thurstonerating, and sport benchmarks for chess,
F1, tennis, sumo, halo2, football -- while main's `winning/ratings` holds two files, although
its `setup.py` already declares `winning.ratings` as a package. Nothing imports `src/` and
`setup.py` packages only `winning.*`, so `src/` is dormant rather than broken. Reconciling it
means editing an overhaul in flight on another machine, so it is recorded here instead.

## Wills et al. corrected and scored into the paper (2026-08-18)

The manuscript said their categorization task has two response categories throughout so the
restriction question does not arise. That was FALSE and is fixed. Experiment 2 disallows one
category for one group of twelve and allows it to another twelve, over identical stimuli.

Reproduced from `wills_twochoice.py`, output now in `results/wills.txt`: 39 cells, 1,560
held-out trials, renorm 0.6839, race 0.6540, gain +0.0299 [+0.0175, +0.0717], excess +0.0314,
p = 0.002. In the paper as a subsection of "The literature that asked first, rescored",
labelled supporting rather than load-bearing because the restriction is between groups with
four participants per disallowed category and the gain sits in two of the three.

The prior-art paragraph now concedes the framing outright: they cite Yellott four times and
state the K >= 3 divergence, so nothing about the Gumbel-point argument is new here. What is
conceded is separated from what is not -- rectangular noise rather than Gaussian, four free
parameters with one set per menu, the renormalization benchmark never computed, nothing held
out.

Scoreboard: thirty-four comparisons, twenty-four wins, five draws, five losses.

## Referee report acted on, and one prediction confirmed against the paper (2026-08-18)

Two mathematical errors corrected, both real.

1. THE SCALE ARGUMENT WAS WRONG. For any location-scale family P_i^S(a;s) = P_i^S(a/s;1), so if
b(p) is the unit-scale calibration then s*b(p) is the scale-s calibration and every restricted
prediction is identical. A common noise scale is a GAUGE, absorbed exactly by the fitted
locations, not approximately. Verified numerically: locations s*a with noise scale s give
pair(0,1) = 0.9076844266 for s = 1, 2 and 5. The appendix paragraph claiming the Gumbel variance
inflation is "invisible in the fit and propagates into the restricted-menu prediction" is
deleted. Consequence for the cumulant expansion: after common centering and dividing by
(1 + eps^2 pi^2/6)^(1/2), the standardized law has skewness O(eps^3) and excess kurtosis
O(eps^4) with no eps^2 term, so the first generic departure from Case V is CUBIC. The reported
local exponents 1.6/2.0/2.3 are dropped as pre-asymptotic or numerical. Everywhere the draft
said "a fitted noise scale would beat both" it now says a fitted SHAPE or mixture parameter.

2. CONCENTRATED SHARES ARE NOT SUFFICIENT. Calibrating Case V to p = (0.90, 0.09, 0.01):
removing the 0.01 gives the favourite 0.9077 against renorm 0.9091, difference 0.0014; removing
the 0.90 gives the surviving 0.09 alternative 0.8028 against renorm 0.9000, difference 0.0972.
The referee's figure reproduces exactly. The condition is that the WITHDRAWN alternatives carry
negligible mass and the leader SURVIVES, not that the shares are concentrated.

3. THE WIN STATISTIC WAS THE WRONG ESTIMAND, and Peter agreed. The scoreboard used excess over
the Luce null, which labelled Wikipedia a win at a raw gain of -0.0001. The table now reports
raw held-out gain and the null tail as separate columns with no win/loss verdict, ordered by
gain, with a family column since rows within a family are not independent. Twenty-eight of
thirty-seven rows favour Gaussian renormalization, stated as a description of the table and not
as an estimated rate.

4. Also fixed: the probit counting argument needed the missing step (the K-1 shares are
exhausted by the K-1 location contrasts, so the K(K-1)/2-1 covariance parameters of the
difference matrix are unidentified for every K >= 3); the boundary questions are no longer
claimed to be answerable from shares alone; a yes to the task-change question now says NEITHER
map applies rather than "renormalization"; near-substitution is stated as an argument for a
similarity-aware model rather than for renormalization; the MIND, ballot and Wikipedia rows are
labelled ecological and the claim that the shared null "makes the comparison fair" is withdrawn;
the favourite-second table is now an aside on sequential heuristics, since removing the observed
winner is conditioning and not withdrawal; lambda is defined in the paper for the first time;
and the softmax claim is weakened to behavioural equivalence under fixed logits.

THE ROULDER PREDICTION. The boundary rule was frozen in a committed draft before the Rouder
chunk data was downloaded. Line length is the canonical unidimensional continuum, so the rule
predicted a loss. It lost: 1,296 cells, 49 subjects, gain -0.0134 [-0.0176, -0.0104], and
-0.0322 in the twelve-to-two condition. That is the largest and best-powered disagreement in the
project and it runs against Gaussian renormalization. It is now in the paper as the third
continuum collection and it is the reason the conclusion claims parity rather than superiority.

## Second-round audit acted on (2026-08-19)

Verified rather than accepted, in each case:

TABLE 1 WAS MIXING POPULATIONS. The consumer rows paired the all-subject gain (+0.0140,
+0.0059) with the forced-choice tail (0.005, 0.129). Confirmed against tab:menus and the text,
which give exp1 tails 0.010 (all) and 0.005 (forced) and exp2 0.199 (all) and 0.129 (forced).
Each experiment now has two rows with gain and tail from the same analysis population. The row
count moves to 39, positives to 30.

YEON-RAHNEV WAS OVERSTATED, AND THE DATA SETTLES IT. Conditions are blocked: 576 blocks in
exp1_trials.csv, exactly one condition each. So subjects knew a two-alternative response was
coming and were trained on it; what was unannounced is the IDENTITY of the surviving pair. The
paper no longer says nothing about the smaller menu could have been anticipated. It says the
design prevents pair-specific advance attention. The conclusion no longer says "the same
observer and stimulus before and after", since the full and restricted responses came from
different trials in different blocks.

TWO CITATIONS WERE WRONG IN THE BIBLIOGRAPHY, both checked at Crossref. Utochkin's coauthors
are Daniil Azarov and Daniil Grigorev, not Sergey Azarov and Nikita Grigorev; the article is
Psychological Science 36(11):831-845. And Meyer-Grant et al. is no longer an OSF preprint: it
is Psychological Review, published online 4 June 2026, doi 10.1037/rev0000615, titled
"Extreme-value signal detection theory for recognition memory: the parametric road not taken".
That is the fourth and fifth wrong reference caught in this project by checking DOIs rather
than search results.

FAVOURITE-SECOND ARITHMETIC. Recomputed from the printed table: the reductions are 4.6, 7.7,
7.2, 13.3 and 9.1 per cent, so "five to twelve" becomes "about five to thirteen".

Also: the cubic claim now carries its smoothness and local-invertibility conditions and notes
that symmetric configurations can cancel the cubic term; Proposition 1 assumes 0 < F(x) < 1 at
every finite x so the reverse hazard is defined, and strictness is stated in terms of H being
non-constant; the covariance sentence is a non-identification claim rather than a global
compensation claim; "needs no additional data" becomes "no additional restricted-menu
observations", with the boundary conditions needing structural information the shares lack;
"a fitted shape parameter beats both" becomes "no worse in sample, held-out improvement needs
separate validation"; the mixture table prints K = 5, A = (0.55, 0.25, 0, -0.25, -0.55),
Gumbel scale 1, 400,000 draws, default_rng(0) and the command that rebuilds it; the stale
"only one where restriction is observed", "a fourth was tried", "twelve of the thirteen
collections" and "four losses are one task" are all corrected; and the unsupported timing
figures are removed.

STILL OPEN. The archive DOI and commit hash are still not printed in the paper, because nothing
is pushed. That is now the only item on the reviewer's list not addressed.

## Verification suite, and what it caught (2026-08-19)

`research/restriction/demo/` holds a JavaScript implementation of both maps from nothing:
Hart's cumulative normal, Simpson quadrature for the contest integral, Newton on the
log-share residual for the inverse. `run_checks.js` asserts one claim per check, twenty in
all, and `index.html` runs them in a browser with an interactive transport widget. All pass.

TWO BUGS IN MY OWN CODE, found by writing the checks. The first erfc gave Phi(1.96) = 0.9621
instead of 0.9750, which would have corrupted everything downstream. And the damped
calibration failed on share vectors with a tiny component, producing six apparent violations
of the contraction claim, all at residuals above 2e-3; the Newton solver reaches machine
precision and the violations vanish. The JS calibration is now more accurate than the lattice
one the analysis uses, whose residual runs to about 1e-3.

`demo/check_tables.py` traces every figure the paper quotes from a run to the output that
produced it, and lists table figures no committed run accounts for. It found three real gaps.

1. THE PAPER'S LARGEST TABLE HAD NO COMMITTED OUTPUT. The twelve-collection held-out table,
thirty figures, was never checked in. `results/heldout_score.txt` now has it and reproduces
every figure exactly.

2. THE TWO DECOMPOSITION TABLES HAD NO SCRIPT AT ALL. They were computed ad hoc for an
earlier draft. `decompositions.py` supplies both. The favourite-second table reproduces,
with Sushi's renormalization figure 0.250 against a printed 0.251, now corrected. The
by-rank table reproduces its shares and its aggregate exactly but NOT its tail: the paper
printed +0.058, +0.101 and +0.171 for ranks eight, nine and ten where the run gives +0.067,
+0.116 and +0.242. My first attempt scored pairs only and missed even the shares; scoring all
subsets, as the paper says it does, matches the shares and the aggregate, so the reproduction
is faithful and the printed tail was stale. Corrected in the paper.

3. ONE FIGURE STILL HAS NO RUN BEHIND IT. The null table's pooled forced-choice gain,
+0.0265. `menus_heldout.txt` has +0.0442 and +0.0140 for the two experiments' forced-choice
subgroups and +0.0100 for all subjects pooled, but no pooled forced-choice figure. Left in
place and flagged rather than quietly altered, since the arithmetic of its excess column is
self-consistent and only the gain itself is untraced.

The audit is now down to two unmatched figures from sixty-six, and both are the derived
excess column, checked by arithmetic instead.

## The auditory branch, incorporated at last (2026-08-19)

None of the auditory notes had reached the paper. Now in Section 2:

CLARKE (1959) IS THE MOTIVATION THE PAPER LACKED. Four signal ensembles, three models, and
the verdict that "the simplified version of the constant-ratio rule and the simplified
version of the theory of signal detectability were both compatible with the data obtained in
the speech experiments". Signal detectability is the Gaussian account, so the two candidates
were run against each other in 1959 and percent correct could not separate them. The same
abstract records the boundary condition: "No model tested was sufficiently complex to account
for data when the sinusoidal signals varied only in amplitude or only in frequency". Both
accounts failed on unidimensional continua at the outset, which is where the tones and the
lines put them.

THE REVERSAL PATTERN IS NOW STATED. Every conformity claim retested with a test that had
power was overturned: Holloway reversed his own 1968 conclusion in 1971, and Morgan's
likelihood-ratio test rejected the rule on both datasets he applied it to, Clarke's and
Egan's. Egan (1957) JASA 29(4):482-489 was missing from the bibliography and is now cited.

## Robinson upgraded, Treisman deliberately not added (2026-08-19)

ROBINSON ET AL. (2023) now carries its numbers. Verbatim quote, the primary analysis holding
each parameter fixed across all m conditions and comparing log likelihood, and the Gaussian
winning in both experiments at t(29) = 4.26 and t(29) = 4.42, p < .001, n = 30 each. The
paragraph then states the distance that remains: m varies by adding fresh items rather than
withdrawing named ones, linear renormalization is never the competitor, and the parameters are
fitted to the data that scores them. Citation re-verified at Crossref: J Math Psychol 117:102805,
doi 10.1016/j.jmp.2023.102805, which is what the bibliography already had. The 137:102805 in a
relayed report was wrong.

TREISMAN AND FAULKNER (1985) IS NOT GOING INTO THE PAPER, and not because it is a threat. The
analysis says it fits d-prime and beta within each m-AFC condition so nothing is held out, never
compares odds between named survivors, restricts no response set, never mentions
renormalization, finds NEITHER parameter invariant, and selects signal detection by a
plausibility argument about the sign of the drift. Its m is confounded with memory load, which is
this project's quality-changing-removal condition. None of fifteen citing works treats it as
having settled the question.

All of that is persuasive and none of it is first-hand: the full text is still unobtained and the
design is known only through the citing literature. Asserting the internals of an unread paper is
the exact failure that produced a fabricated title earlier in this project, so the paper says
nothing about it. The bibliography entry was already dropped when it turned up uncited. If a
referee raises it, the notes hold the answer and the answer can then be written from the source.
