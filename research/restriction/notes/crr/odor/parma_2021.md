# Parma et al. 2021 — SCENTinel 1.0 (4 options, then the 3 remaining)

## Citation
Parma V, Hannum ME, O'Leary M, Pellegrino R, Rawson NE, Reed DR, Dalton PH. "SCENTinel 1.0:
Development of a Rapid Test to Screen for Smell Loss." *Chemical Senses* 2021;46:bjab012.
doi:10.1093/chemse/bjab012. PMC8083606. (Verified via Europe PMC core record.)

## Domain and stimuli
Olfaction, rapid single-odor screening test on a printed card (Monell Chemical Senses Center).
Measures detection, intensity, pleasantness, and identification. Self-administered, deployed
remotely at large scale across the SCENTinel 1.0 / 1.1 / 2.0 series.

## Master response set and restricted response sets (nested, overlapping, or a relabelling)
**NESTED, and it is the exact structure sought — the only live instance found.**
Master set S = 4 labeled options (word plus picture) on the first identification attempt. If the
participant is wrong, they get a second attempt over **the 3 remaining options**, i.e.
**T = S \ {first choice}**. Same odor, same subject, seconds apart, and both responses are recorded.

Verified from the Methods, quoted:
> "select the best verbal and visual label for the odor among 4 options provided"
> "Participants who gave an incorrect response to (c) were instructed to try again to identify the
> odor, this time among the 3 remaining options."

Note this is a *subject-determined* restriction: T depends on what the participant picked first, so
it is a nested menu but not an experimenter-randomized one. That is an asset (the removed element is
known per trial) and a liability (selection on the first response), and any analysis has to condition
on the first choice.

## What numbers are printed or deposited (which tables/files, counts or proportions, per subject or pooled)
**Both attempts are recorded in the authors' data; almost nothing is published.**
- Table 1 of the 2021 paper gives an accuracy matrix with separate "First attempt" and "Second
  attempt" outcomes — pooled, not per descriptor.
- The supplementary .docx carries aggregate response-pattern tables only.
- Successor papers print occasional single distractor shares in prose rather than tables — see
  `hunter_2024.md` for the clearest example.

No trial-level deposit exists. Every advertised repository was checked and each one fails:
- OSF `g89dq` → HTTP 401, private.
- OSF `5d7kx` → HTTP 401, private.
- DOI 10.17605/OSF.IO/5R9JB → resolves (`api.osf.io/v2/registrations/5r9jb/`, public) but contains
  exactly one file, `SCENTinel algorithm 1.0.pdf`.
- Public registrations `cr7ps`, `us2v4`, `twu4e` → PDFs and R/Rmd scripts only, no data.
- No SCENTinel repository exists on GitHub.

## Access (a DIRECT url you fetched; open, paywalled, or Wayback-only)
https://pmc.ncbi.nlm.nih.gov/articles/PMC8083606/ — fetched, **open access**, full methods readable.
Successor papers, all open: PMC9935080 (SCENTinel 1.1, *Chem Senses* 2023), PMC11041634
(*Front Public Health* 2024, n=1,979), PMC11519045 (*Chem Senses* 2024).

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)
**Unusable now; highest-value email in the whole search.**
Nothing to digitize — the paired first/second choice records are on Monell's disks and were never
deposited. But the data already exists, at scale: cumulative n across the SCENTinel papers is well
over 5,000, every participant who erred on attempt one contributes a nested (4 → 3) pair on the same
odor, and both responses were logged by design.

Ask, in order: Sarah Hunter, Danielle Reed, Pamela Dalton (Monell); Valentina Parma (now Wageningen).
Note the useful coincidence that Dalton and Parma are also authors on `jaen_2024.md` and
`tolomeo_2026.md` respectively, so a single Monell contact could unlock three of the datasets in this
directory.

## What the authors concluded, quoted verbatim where possible
The 2021 paper's conclusions concern screening validity, not menu size — the second attempt exists to
improve sensitivity, and the authors never analyze it as a choice-set manipulation. From the abstract
framing, SCENTinel "rapidly screens for smell loss" and the multi-attempt scoring improves
discrimination of anosmic from normosmic respondents. The nested structure is incidental to their
purpose and entirely unexploited, which is precisely why the raw records are worth requesting.
