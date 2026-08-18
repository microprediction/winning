# Albantakis, Branzi, Costa & Deco (2012) — 2- vs 4-alternative random-dot motion

**Status: trial-level data is NOT public. Nothing was downloaded. This directory
contains no data.**

## Citation (exact)

Albantakis, L., Branzi, F. M., Costa, A., & Deco, G. (2012). A Multiple-Choice
Task with Changes of Mind. *PLoS ONE*, **7**(8), e43131.
https://doi.org/10.1371/journal.pone.0043131

- Publisher page (HTTP 200): https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0043131
- PubMed Central (HTTP 200): https://pmc.ncbi.nlm.nih.gov/articles/PMC3420917/
- Full JATS XML: https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0043131&type=manuscript
- Corresponding author: Larissa Albantakis, albantakis@wisc.edu
  (then Universitat Pompeu Fabra, Barcelona / Dept. of Psychiatry, University of
  Wisconsin–Madison)

This is the paper matching the requested description. Note the title is "A
Multiple-Choice Task with **Changes of Mind**" — the headline measure is
change-of-mind trajectories, not choice frequencies.

## The design matches the request exactly

Verbatim from the Methods (Stimuli / Experimental Sessions):

> "We used a set of eight different coherence levels (0%, 3.2%, 6.4%, 12.8%,
> 25.6%, 51.2%, 76.8% and 100%). The R-targets were constructed as yellow circles
> (diameter = 2.0°) located at the corners of a virtual square around the central
> fixation-mark (edge length 28°, and thus 19.8° distance to the center). The
> location of the R-targets indicated the possible directions of coherent motion
> in each trial. They could appear either: in each of the four corners of the
> virtual square (4-choice trials), or in just two of the four corners (2-choice
> trials). Figure 1B illustrates the R-target locations in the 4-choice trials
> (top right) and three of the six possible R-target combinations for 2-choice
> trials. The R-targets remained present on the screen until the end of the
> trial."

> "Our participants underwent four experimental sessions of 30 minutes each, all
> in the same day, separated by a time interval of two hours. In the first three
> sessions, we tested the participants on the combined 2- and 4-alternative task,
> explained above. Each of these sessions consisted of a total of 348 trials: 232
> trials with two choice alternatives and 116 4-choice trials, presented in
> random order. In half of the 2-choice trials the R-targets were located at
> opposite screen corners such as shown in the upper left panel of Figure 1B. In
> the other half of the 2-choice trials the R-targets were located at the same
> side (up, down, left, or right of the screen). The eight coherence levels were
> presented 32 (2-alternative) or 16 (4-alternative) times each, except for 0%
> which was only presented eight or four times, respectively. The 0% coherence
> level was presented less often to avoid frustrating participants with unsolvable
> trials. For 0% coherence the 'correct' target was defined randomly."

Other design facts:

- 14 healthy young adults (10 female; mean age 22, range 19–27), right-handed,
  normal vision. **One excluded** for accuracy below 65%, so analyses use 13.
- Responses were made by moving a mouse pointer from a central start position to
  the chosen yellow R-target; full mouse trajectories were recorded (that is how
  changes of mind were detected).
- Sessions 1–3 are the intermixed 2-/4-choice task. **Session 4 is a different
  control** ("2-Top control") replicating the Resulaj et al. binary layout with
  both targets at the top; it is not part of the 2-vs-4 comparison.
- Supporting Information: Figures S1–S4 and Tables S1–S2 only, all `.pdf`
  (S1 = choice behavior in the 2-top control; S2 = race-model parameter
  variation; S3 = attractor model vs change frequency distributions; S4 =
  diffusion-model fit; S1/S2 tables = model summary and simulation parameters).
  **No data file among them.**

So the *design* does distinguish which two of the four corners were available on
every 2-choice trial, and all six pairs occur. Whether the *recorded files*
preserve that per trial cannot be verified, because those files are not released.

## Availability: what was checked (2026-08-17)

| Check | Result |
|---|---|
| PLOS Data Availability statement | **None.** The article predates PLOS's March 2014 mandatory data-availability policy. The JATS XML contains no `Data Availability` section and no occurrence of "availability", "Dryad", "figshare", "repository", or "request". |
| PLOS Supporting Information | 6 items, all PDF figures/tables (`pone.0043131.s001`–`s006`). No `.csv`, `.mat`, `.txt`, `.xls` or archive. Confirmed via the PLOS Solr API (`api.plos.org/search`, field `supporting_information`) and the article XML. |
| Dryad | `datadryad.org/api/v2/search?q=Albantakis` — 1 unrelated hit (a different Dryad dataset that merely cites an "Albantakis"). No deposit for this study. |
| Zenodo | `zenodo.org/api/records?q=Albantakis` — no dataset for this study. |
| DataCite | `api.datacite.org/dois?query=Albantakis` — 102 hits, none a behavioural dataset for this paper. |
| OSF | No project or preprint found for this study. |
| GitHub | `github.com/albantakis` has 14 public repos, all IIT / PyPhi / animat / TI-toolbox work. No repo for this paper; GitHub repo search for it returns 0 results. |
| Datasets citing the paper | OpenAlex work `W2044859786` (`https://api.openalex.org/works/doi:10.1371/journal.pone.0043131`): 29 citing works, **0** of type `dataset`. No secondary release found. |

**Conclusion: the trial-level data for Albantakis et al. (2012) is not publicly
available.** There is no deposit, no supplementary data file, and no data
availability statement. The only route to it is a direct request to Larissa
Albantakis (albantakis@wisc.edu). Given the paper is from 2012 and the data was
collected at Universitat Pompeu Fabra with mouse-trajectory recordings, recovery
is not guaranteed.

## What a request would need to ask for, for a restriction test

To use this dataset for a restriction/IIA test the per-trial record must contain,
at minimum:

1. Participant ID and session number (sessions 1–3 only; exclude session 4).
2. `n_alternatives` (2 or 4).
3. **The set of available R-target corners on that trial** — i.e. two of
   {up-left, up-right, down-left, down-right} on 2-choice trials, all four on
   4-choice trials. This is the load-bearing field and is exactly what is
   unverifiable from the publication. The published figures only bin 2-choice
   trials into "opposite corners" vs "same side", so even the aggregate reported
   in the paper does not resolve all six pairs.
4. **Which corner was chosen** (the final R-target reached), and separately the
   initially-headed-for corner, since changes of mind are the paper's focus and
   the initial and final choice can differ.
5. Motion coherence level and the true (correct) direction.
6. Ideally the mouse trajectory, RT, and change-of-mind flag.

If (3) and (4) are both present, the six 2-alternative menus over a fixed
four-item set plus the full four-item menu give a genuinely well-powered
restriction design — better than the Comay dud data, because all four items are
fixed spatial labels rather than trial-varying size roles. That is the reason to
try the email.
