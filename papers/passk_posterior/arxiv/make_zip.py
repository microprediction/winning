#!/usr/bin/env python3
"""Rebuild the arXiv upload: paper.tex alone (bibliography inline, no figures)."""
import zipfile
from pathlib import Path
here = Path(__file__).resolve().parent
with zipfile.ZipFile(here / "passk_arxiv.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write(here.parent / "paper.tex", "paper.tex")
print("wrote", here / "passk_arxiv.zip")
