# Rouder laboratory "chunk" experiments, twelve line lengths, nested response sets

Unpublished laboratory data, Perception and Cognition Laboratory, University of Missouri.
Public in `PerceptionCognitionLab/data0` under `1dMemory/chunk`, no licence file and no
README. No associated publication was found: Crossref searches on Rouder plus chunking or
absolute identification return only the 2001 Psychological Science and 2004 PBR papers,
neither of which is this design. Git history for the path is two commits, "adding 1dMemory
expts for Richard" (2017-04-02) and "adding richard's data" (2019-03-07), so it is probably
legacy data belonging to Richard Morey. Treat authorship as unattributed until someone in
the laboratory confirms it.

Downloaded 2026-08-18, no login. `MANIFEST.tsv` records every file with its upstream git
blob SHA. Compiled artefacts (`.EXE`, `.OBJ`, `a.out`) were discarded; the C sources are
kept because the design has to be read off them.

## Why it is the strongest instrument in the corpus

Subjects learn a fixed number for each of twelve line lengths and identify them under
feedback. From `c1instruct.txt`, verbatim:

> "The number assignment for each line is constant throughout the experiment. Some of the
> blocks will have all 12 line lengths. Others will have fewer than 12. Before each block,
> you will be given the line lengths that will be shown and their corresponding number."

So the logged stimulus and response are indices into the master set, not within-block
ranks, and the same subject supplies both the full-menu distribution and the restricted one.
Confirmed independently from `C2R/C2.C`, where `blocktypestims` holds global indices.

| Sub-experiment | Master | Restricted sets | Subjects |
|---|---|---|---|
| `c0` cond A, D | 12 | {0,1,2,3}, {4,5,6,7}, {8,9,10,11}, {0..7}, {4..11} | 25 + 10 |
| `c0` cond C | 12 | twelve named pairs, identical across subjects | 14 |
| `c0` cond B | 12 | none, all-twelve control | 15 |

Restricted blocks are bracketed by full-twelve blocks before and after, which is the
bracket that keeps practice from loading onto the restricted blocks.

## Result: the clearest loss for Gaussian renormalization in the project

`research/restriction/rouder_chunk.py`, output in `results/rouder_chunk.txt`. 1,296 cells
over 49 subjects. Linear normalization predicts better in every split, and the subject
bootstrap excludes zero everywhere.

    all restricted blocks   1296 cells  renorm 0.7846  race 0.7980  gain -0.0134  [-0.0176, -0.0104]
    twelve to two           328 cells   renorm 0.1756  race 0.2079  gain -0.0322  [-0.0371, -0.0279]
    twelve to four          416 cells   renorm 0.7820  race 0.7910  gain -0.0090  [-0.0122, -0.0058]
    twelve to eight         552 cells   renorm 1.1483  race 1.1540  gain -0.0057  [-0.0079, -0.0029]

This was predicted before the run. Line length is the canonical unidimensional continuum of
the absolute-identification literature, and the paper's boundary rule, frozen in a committed
draft before the data was downloaded, says Gaussian renormalization loses where the
alternatives lie on a perceptual continuum. It is the third such collection after the tone
matrices and the Getty condition whose survivors are mutual confusions, and the loss is
largest where the menu is cut hardest, from twelve to two.

## Not scored here

`c2` and `C3` use seven lines with a crossed spacing manipulation, and `C2R` has a
different ten-column format with task-phase and subset indices. Both are usable and are
left for a later pass. Condition B is an all-twelve control with no restriction.
