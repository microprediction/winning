## Citation

Unpublished lab data, Perception & Cognition Lab (Jeff Rouder), University of Missouri.
Repository `PerceptionCognitionLab/data0`, path `1dMemory/chunk`. No associated
publication found: Crossref searches on Rouder + chunking/absolute identification return
only the 2001 Psych Science and 2004 PBR papers, neither of which is this design. Git
history for the path is two commits only, "adding 1dMemory expts for Richard" (2017-04-02)
and "adding richard's data" (2019-03-07), so it is probably archived legacy data belonging
to Richard Morey. Treat authorship as unattributed until someone in the lab confirms it.

## Domain and stimuli

Visual line length, absolute identification with feedback. Four sub-experiments.
`c0`: 12 lines, lengths {23,32,43,57,73,93,116,143,174,210,251,298} px (from `C1.C`).
`C2R`, `c2`, `C3`: 7 lines, with a crossed spacing manipulation — `lengthwide` =
{23,39,99,123,148,230,260} (clumped into 2/3/2 groups) versus `lengthnarrow` =
{57,77,99,123,148,174,201} (near-even). Responses on the top keyboard row, 1..9 then
`-` for 11 and `=` for 12.

## Master response set and restricted response sets (nested, overlapping, or a relabelling)

**Genuinely nested, and the labels are global.** Verified two independent ways.

From `C2R/C2.C`:

    int blocktypestims[4][7]={{0,1},{2,3,4},{5,6},{0,1,2,3,4,5,6}};
    int blockstimnums[4]={2,3,2,7};
    ...
    stimulus=blocktypestims[blocktypes[b]-1][stim[t]];

so the logged stimulus code is the index into the master array, not a within-block rank.

From the participant instructions (`c1instruct.txt`), verbatim:

> "The number assignment for each line is constant throughout the experiment. Some of the
> blocks will have all 12 line lengths. Others will have fewer than 12. Before each block,
> you will be given the line lengths that will be shown and their corresponding number."

Empirically confirmed by tabulating distinct stimulus and response codes per block:

| Sub-exp | Master set | Restricted sets | Subjects | Trials/subject |
|---|---|---|---|---|
| `c0` cond A (`C1AS*`), cond D (`C1DS*`) | 12 | {0-3},{4-7},{8-11} and {0-7},{4-11} | 25 + 10 | 880 |
| `c0` cond C (`C1CS*`) | 12 | 12 distinct **pairs**: {0,2},{1,3},{2,4},{3,5},{4,6},{5,7},{6,8},{7,9},{8,10},{9,11},{0,10},{1,11} | 14 | 880 |
| `c0` cond B (`C1BS*`) | 12 | none — all-12 control | 15 | 880 |
| `C2R` | 7 | {0,1},{2,3,4},{5,6} | 47 | 782 |
| `c2` (`CHN/CHW/CNN/CNW`) | 7 | {0,1},{2,3,4},{5,6} | 13/10/12/12 | 782 |
| `C3` (`C3_*`, `C3R_*`) | 7 | {2,3,4} only, **order counterbalanced** | 8 + 4 | 756 |

In every restricted block the set of observed responses equals the set of offered global
labels — no leakage, no rank recoding. The pair list in condition C is identical across all
14 subjects.

## What numbers are printed or deposited (which tables/files, counts or proportions, per subject or pooled)

Raw trial-level data, one plain-text file per subject, space-delimited, no header.

`c0` and `c2`/`C3` are 7 columns: `sub blk trl bt stim resp RT_ms`.
`C2R` is 10 columns: `sub ch wd blk trl bt set stim resp RT_ms` where `bt` is the task
phase (1 warmup, 2 pretest, 3 training, 4 posttest) and `set` indexes which subset.

Verified cell counts, `C2R/C2RS03`: pretest full-7 n=140; training {0,1} n=144,
{2,3,4} n=144, {5,6} n=144; posttest full-7 n=140.

Verified `c0/C1AS16`: 22 blocks x 40 trials. Six full-12 blocks (bt0,bt1), then
{0-3},{4-7},{8-11} three times each (bt2,bt3,bt4, 120 trials per subset), then {0-7} and
{4-11} twice each (bt5,bt6, 80 trials per subset), then three full-12 posttest blocks (bt7).

Verified `c0/C1CS12`: six full-12 blocks, then 13 pair blocks of 40 trials, then three
full-12 posttest blocks. So ~360 full-set trials and 520 pair trials per subject.

Per subject, not pooled. Roughly 174 real subjects across the four sub-experiments once
`TEST*` files are excluded.

## Access (a DIRECT url you fetched; open, paywalled, or Wayback-only)

**Open**, no login. Fetched and confirmed HTTP 200:

- Tree: `https://github.com/PerceptionCognitionLab/data0/tree/master/1dMemory/chunk`
- Instructions: `https://raw.githubusercontent.com/PerceptionCognitionLab/data0/master/1dMemory/chunk/c1instruct.txt`
- Example data: `https://raw.githubusercontent.com/PerceptionCognitionLab/data0/master/1dMemory/chunk/C2R/C2RS03`
- Example data: `https://raw.githubusercontent.com/PerceptionCognitionLab/data0/master/1dMemory/chunk/c0/C1AS16`
- Stimulus definitions: `.../chunk/C2R/C2.C`, `.../chunk/c0/C1.C`

Repo metadata: public, last pushed 2026-05-14, **no LICENSE file**. Clarify reuse terms
before publishing anything derived from it.

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**Usable now.** The single best source found. It is the only one with raw trials, a
master set, nested restrictions, and global labels all at once, and the only one with
enough subjects for per-subject fits.

Two design cautions.

1. Order is fixed in `c0`, `C2R` and `c2`: full set, then subsets, then full set. Practice
   is therefore confounded with restriction. Mitigations: use the pre/post full-set blocks
   to bracket the baseline, use the all-12 control condition B as the practice-only
   reference, and use `C3` (12 subjects) where subset-first versus full-first is
   counterbalanced.
2. Condition C is the sharpest instrument for a ratio-rule test, because it yields binary
   choice over the same labelled pair and a 12-alternative choice from the same subject.
   Note the pairs are mostly gap-2 neighbours plus two long-distance pairs ({0,10},{1,11}),
   so near-substitute and far-apart cases can be separated — relevant to the standing
   caution in the branch README about order reversal on removal.

## What the authors concluded, quoted verbatim where possible

No publication and no README, so there is no author conclusion to quote. The design
intent is legible only from the code and the participant instructions, both quoted above.
The `debrief.tex` file in the `chunk` directory may state the hypothesis; not yet read.
