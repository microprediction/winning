# Venue notes

## Primary

- **JCGS** (Journal of Computational and Graphical Statistics) — target.
  Submission build is `scalable-share-calibration-jcgs.pdf` (double-spaced; regenerate by flipping \jcgsfalse to \jcgstrue in the main tex, three pdflatex passes), supplement is
  `supplement_v1.1.0.zip`.
- **SSRN** — preprint of `scalable-share-calibration.pdf`, abstract id
  7307363, **doi:10.2139/ssrn.7307363** (registered in Crossref; cite
  this as the method reference, e.g. the `mvtnormfast` DESCRIPTION
  does). JCGS permits preprint posting of the author's original
  manuscript.
  - *Open (confirmed 2026-08-29, live record inspected):* the posted
    version **is stale** — 41 pages posted 20 Aug against 44 in the
    current build, with seven commits landing 24 Aug (prior-art
    citations, N=2 inversion fix). Two fields need revising:
    - **the PDF** (should also fix the 0-references: 109 `\bibitem`s
      under a standard "References" heading that `pdftotext` extracts
      cleanly);
    - **the abstract**, which matters more — the posted text is the
      pre-correction version and overstates. It says the shares
      "carry log-share errors" where the corrected text says they
      "agree with higher-resolution references of the same
      construction" (self-convergence, named as such); it omits the
      independent RQMC check sentence; and it calls the 140x
      comparator "the nearest factor-aware competitor" where the
      corrected text names it precisely as "a per-alternative
      factor-conditioned RQMC baseline … on the reported hardware …
      because it performs N separate integrals".

    Paste-ready text and the full checklist: `~/Downloads/ssrn/`.
    Record stats at inspection: 18 downloads, 51 abstract views.

## Fallbacks worth considering

Computational Statistics & Data Analysis; Journal of Statistical
Software (if reframed around the package); Transportation Research
Part B (the demand-inversion audience); Journal of Choice Modelling.

## Not a fallback: Epistora Publications

An unsolicited invitation arrived 2026-08-20 from "Epistora Journal of
Artificial Intelligence and Intelligent Systems"
(ai@epistorapublications.com, "Anna Grace, Managing Editor"), inviting
a submission by 2026-09-15.

Recorded here because Peter asked for the note, with the caveat that
every marker of a predatory operation is present:

- unsolicited, with flattery ("in recognition of your expertise") but no
  engagement with any actual paper of his;
- an article-type menu including **Case Reports**, which is medical
  boilerplate and meaningless for an AI journal;
- an offer to shop the manuscript across an unnamed "portfolio" until
  some journal takes it, which is the opposite of editorial fit;
- a submission deadline unattached to any special issue;
- no editorial board, impact record, or indexing named; the domain has
  no publishing track record.

Publishing here would not constitute peer review, would likely carry an
article processing charge, and would attach a predatory venue to the
paper's permanent record — including to the `winning` package that
cites it. If JCGS declines, the fallbacks above are real journals with
real referees; this is not a last resort but a trap.
