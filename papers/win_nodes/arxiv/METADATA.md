# arXiv submission metadata

Upload: arxiv/win_nodes_arxiv.zip (paper.tex with the bibliography inline; the one
figure is TikZ). arXiv compiles with pdflatex; the packages used (geometry, amsmath,
amssymb, amsthm, booktabs, graphicx, tikz, microtype, natbib, hyperref) are all in its TeX
Live. Rebuild the zip with `python3 arxiv/make_zip.py` after editing paper.tex or the
figure.

Title: Win Nodes for Factor-Structured Gaussian Bayes Nets
Authors: Peter Cotton
Primary category: stat.ML
Cross-list: stat.CO, cs.LG, cs.AI
Comments: 16 pages, 1 figure. Code and experiment scripts: github.com/microprediction/winning
(papers/win_nodes, research/win_nodes, research/rs_crn).
License: arXiv non-exclusive license, or CC BY 4.0 if the paper is to be reused freely.
Abstract: paste arxiv/abstract.txt (plain text; arXiv accepts $...$ math).

This paper supersedes the working drafts papers/exact_pom ("Deterministic Probabilities
for Thompson Sampling and Entropy Search") and the earlier papers/win_nodes draft.

After it is announced: put the arXiv id in microprediction/home papers.json (replacing
the exact_pom and win_nodes entries), and in papers/README.md here.
