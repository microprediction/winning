from __future__ import annotations

import numpy as np

from .density import Density, _pdf_from_cdf

EPS = 1e-18
DEL = 1e-12


def _conditional_win_draw_loss(
    pdfA: np.ndarray, pdfB: np.ndarray, cdfA: np.ndarray, cdfB: np.ndarray
):
    winA = pdfA * (1.0 - cdfB)  # X < Y
    draw = pdfA * pdfB  # X == Y
    winB = pdfB * (1.0 - cdfA)  # Y < X
    return winA, draw, winB


def _winner_of_two(
    dA: Density,
    dB: Density,
    multA: np.ndarray | None = None,
    multB: np.ndarray | None = None,
) -> tuple[Density, np.ndarray]:
    cA = dA.cdf()
    cB = dB.cdf()
    cMin = 1.0 - (1.0 - cA) * (1.0 - cB)
    pdfMin = _pdf_from_cdf(cMin)
    out = Density(dA.lattice, pdfMin)

    L = dA.lattice.L
    if multA is None:
        multA = np.ones(2 * L + 1, dtype=float)
    if multB is None:
        multB = np.ones(2 * L + 1, dtype=float)

    wA, dr, wB = _conditional_win_draw_loss(dA.p, dB.p, cA, cB)
    numer = wA * multA + dr * (multA + multB) + wB * multB + EPS
    denom = wA + dr + wB + EPS
    mult = numer / denom
    return out, mult


def winner_of_many(densities: list[Density]) -> tuple[Density, np.ndarray]:
    if len(densities) == 0:
        raise ValueError("winner_of_many requires at least one density.")
    d = densities[0]
    L = d.lattice.L
    mult = np.ones(2 * L + 1, dtype=float)
    for d2 in densities[1:]:
        d, mult = _winner_of_two(d, d2, mult, np.ones_like(mult))
    return d, mult


def get_the_rest(
    density: Density,
    densityAll: Density | None,
    multiplicityAll: np.ndarray,
    cdf: np.ndarray | None = None,
    cdfAll: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if cdf is None:
        cdf = density.cdf()
    if cdfAll is None:
        if densityAll is None:
            raise ValueError("Need densityAll or cdfAll.")
        cdfAll = densityAll.cdf()
    pdf = _pdf_from_cdf(cdf)
    pdfAll = _pdf_from_cdf(cdfAll)

    S_all = 1.0 - cdfAll
    S_self = 1.0 - cdf
    S_rest = (S_all + EPS) / (S_self + DEL)
    cdfRest = 1.0 - S_rest
    cdfRest = np.maximum.accumulate(cdfRest)
    pdfRest = _pdf_from_cdf(cdfRest)

    m = multiplicityAll
    f1 = pdf
    m1 = 1.0

    numer = m * f1 * S_rest + m * (f1 + S_self) * pdfRest - m1 * f1 * (S_rest + pdfRest)
    denom = pdfRest * (f1 + S_self) + EPS
    mult_left = (EPS + numer) / denom

    T1 = (S_self + EPS) / (f1 + DEL)
    Trest = (S_rest + EPS) / (pdfRest + DEL)
    mult_right = m * Trest / (1.0 + T1) + m - m1 * (1.0 + Trest) / (1.0 + T1)

    k = int(np.argmax(f1))
    mult = mult_left.copy()
    mult[k:] = mult_right[k:]
    # multiplicity is a count-like quantity; enforce non-negativity
    mult = np.maximum(mult, 0.0)
    return cdfRest, mult


def expected_payoff_with_multiplicity(
    density: Density,
    densityAll: Density,
    multiplicityAll: np.ndarray,
    cdf: np.ndarray | None = None,
    cdfAll: np.ndarray | None = None,
) -> np.ndarray:
    cRest, mRest = get_the_rest(density, densityAll, multiplicityAll, cdf=cdf, cdfAll=cdfAll)
    pdf = density.p if cdf is None else _pdf_from_cdf(cdf)
    win, draw, _ = _conditional_win_draw_loss(pdf, _pdf_from_cdf(cRest), density.cdf(), cRest)
    # multiplicity must be finite and >= 0; clamp small negative numerical noise
    mRest = np.maximum(mRest, 0.0)
    if not np.all(np.isfinite(mRest)):
        raise ValueError("Multiplicity contains non-finite values.")
    return win + draw / (1.0 + mRest)
