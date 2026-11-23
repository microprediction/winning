"""Paste-ready arXiv abstracts are metadata, not manuscript (#951).

The Abstract field has no bibliography, so a natbib or BibTeX citation
command reaches the public abstract page, the announcement email and the
API record as raw markup. arXiv accepts $...$ math and TeX accent commands
there, caps the field at 1,920 characters, and asks for ASCII.
"""
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
ABSTRACTS = sorted(ROOT.glob("papers/*/arxiv/abstract.txt"))


def test_there_are_abstracts_to_check():
    assert ABSTRACTS, "no papers/*/arxiv/abstract.txt found: the glob is stale"


@pytest.mark.parametrize("path", ABSTRACTS, ids=lambda p: p.parts[-3])
def test_abstract_is_self_contained_metadata(path):
    text = path.read_text()
    cites = re.findall(r"\\(?:cite[a-zA-Z]*|ref|eqref|autoref|label)\b", text)
    assert not cites, f"{path}: bibliography-dependent commands {sorted(set(cites))}"
    assert text.isascii(), f"{path}: non-ASCII characters; use TeX accents"
    assert len(text.strip()) <= 1920, f"{path}: {len(text.strip())} characters > 1920"
