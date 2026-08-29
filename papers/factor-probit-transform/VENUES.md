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
  - *Open:* SSRN shows **0 references** for the record although the
    manuscript carries 109 `\bibitem`s under a standard "References"
    heading that `pdftotext` extracts cleanly. Most likely the posted
    PDF is stale — the same staleness that had the JCGS build seven
    commits behind. Re-upload the current build and recheck.

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
