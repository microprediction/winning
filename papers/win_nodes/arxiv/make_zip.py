#!/usr/bin/env python3
"""Rebuild the arXiv upload: paper.tex (bibliography inline) and nadal.pdf."""
import zipfile
from pathlib import Path
here = Path(__file__).resolve().parent
with zipfile.ZipFile(here / "win_nodes_arxiv.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write(here.parent / "paper.tex", "paper.tex")
    z.write(here.parent / "nadal.pdf", "nadal.pdf")
print("wrote", here / "win_nodes_arxiv.zip")
