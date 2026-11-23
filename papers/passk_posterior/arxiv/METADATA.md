# arXiv submission metadata

Upload: arxiv/passk_arxiv.zip (one .tex file with the bibliography inline, no figures). arXiv
compiles with pdflatex; the packages used (geometry, amsmath, amssymb, amsthm, booktabs,
microtype, natbib, hyperref) are all in its TeX Live. Rebuild the zip with
`python3 arxiv/make_zip.py` after editing paper.tex. The refs/ folder holds a third-party PDF
for reference and is not part of the upload.

Title: A Posterior Predictive for Pass@k: Established Empirical Bayes, Measured on Released Rollouts
Authors: Peter Cotton
Primary category: cs.LG
Cross-list: cs.CL, stat.ML
Comments: 7 pages. An independent evaluation of the posterior-predictive pass@k estimator of
Verine et al. on nine thousand released rollouts with an extrapolative split.
License: arXiv non-exclusive license, or CC BY 4.0 if the note is to be reused freely.
Abstract: paste arxiv/abstract.txt (plain text; arXiv accepts $...$ math).

SSRN, if posted the same day: same title and abstract; keywords: pass@k, empirical Bayes,
posterior predictive, language model evaluation, repeated sampling, coverage; discipline:
computer science (machine learning), statistics.

After it is announced: put the arXiv id in microprediction/home papers.json, on the
brownianbandit papers page (which serves the PDF), and drop any pending line in posted.json.
