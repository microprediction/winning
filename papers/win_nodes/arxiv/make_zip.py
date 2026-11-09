#!/usr/bin/env python3
"""Rebuild the arXiv upload: paper.tex (bibliography inline; no figure files)."""
import zipfile
from pathlib import Path
here = Path(__file__).resolve().parent
with zipfile.ZipFile(here / "win_nodes_arxiv.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write(here.parent / "paper.tex", "paper.tex")
print("wrote", here / "win_nodes_arxiv.zip")
