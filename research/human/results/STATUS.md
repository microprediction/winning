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
largest gain; all inputs committed under `research/human/data` at 1.9MB so nothing
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
- Nothing is pushed. The referee could not see `research/human` on main for that
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

research/human/tones.py on the digitized Stewart, Brown and Chater matrices. Calibrate on the
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
