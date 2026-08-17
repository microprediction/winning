# Townsend & Landon (1982) — digitized confusion matrices

Townsend, J. T., & Landon, D. E. (1982). An experimental and theoretical investigation
of the constant-ratio rule and other models of visual letter confusion.
*Journal of Mathematical Psychology*, **25**, 119–162.

## Source

- URL fetched: `https://web.archive.org/web/20160916190658if_/http://www.indiana.edu/~psymodel/papers/towlan82.pdf`
- HTTP 200, 2,699,621 bytes, 44 pages, PDF 1.3, produced by "Acrobat 4.0 Capture
  Plug-in for Windows" (i.e. a scan with an OCR text layer, not born-digital text).
- Extraction: `pdftotext -layout` and `pdftotext -bbox` for the text layer;
  `pdftoppm -r 360` crops read visually to confirm every cell.

| Subject file id | Table | PDF page | Journal page | Subject label in paper |
|---|---|---|---|---|
| `s1` | Table 1 | 24 | 142 | D. X. |
| `s2` | Table 2 | 25 | 143 | M. X. |
| `s3` | Table 3 | 26 | 144 | G. X. |
| `s4` | Table 4 | 27 | 145 | A. X. |

Each table holds four confusion matrices: a 5×5 master over {A,E,F,H,X}, a 4×4 over
{A,E,F,H}, and 3×3s over {A,E,X} and {F,H,X}. Rows are presented stimuli, columns are
spoken responses. Each matrix came from a separate block of presentations.

## Quartered-cell convention — CONFIRMED

The footnote beneath each table reads:

```
OBT = obtained response proportions.       WSCM = weak similarity choice model predicted proportions.
CRR = constant-ratio rule predicted        SSCM = strong similarity choice model predicted proportions.
      proportions.
```

and the body text (journal p. 142) states: "Each confusion cell in Tables 1–4 has been
quartered. The obtained response proportions appear in the upper left quadrant of each
cell. Predicted response proportions for the WSCM, SSCM, and CRR appear in the upper
right, lower right, and lower left quadrants of each cell, respectively."

So the assumed layout **matched**:

```
 upper-left  = OBTAINED   <-- the only quadrant digitized here
 upper-right = WSCM
 lower-left  = CRR
 lower-right = SSCM
```

Two documented exceptions, both in the 5×5 master matrix only:

- No CRR entry exists for the master matrix (the CRR predicts subsets *from* the
  master), so the lower-left quadrant is empty there.
- WSCM and SSCM coincide for the master matrix, and the single shared prediction is
  printed in the lower-right (SSCM) quadrant. Hence master-matrix cells show two
  numbers, not four, and the second one is a prediction, not data.

This was independently confirmed geometrically from `pdftotext -bbox`: obtained values
sit at x ≈ 95/162/229/295/361 pt (columns A/E/F/H/X), the upper-right predictions at
x ≈ 119/186/252/318/384 pt, and the lower-left/lower-right predictions at the same two
x-bands one line lower.

## Files

- `townsend1982_s<N>_<setname>.csv` — **counts out of 240** (integers). These are the
  analysis-ready files.
- `townsend1982_s<N>_<setname>_prop.csv` — the printed obtained proportions verbatim,
  as a provenance record.

`<setname>` ∈ {`AEFHX`, `AEFH`, `AEX`, `FHX`}; 16 of each; 32 CSVs total.
First column header `stimulus`, then one column per response letter.

## Why counts are recoverable

Method (journal p. 141): 15 trials per letter per block per session, 16 sessions,
"When summed over sessions, each letter appeared 240 times for each block in which it
appeared." So **every row of every matrix has n = 240**, and since 1/240 = .004167 is
larger than the .0005 rounding granularity of a 3-decimal proportion, each printed
proportion determines its count uniquely. Counts were recovered as the integer vector
summing to 240 that minimises the maximum deviation from the printed proportions; in
every resolved row that vector is unique.

## Verification results

Two checks on all 60 rows (4 subjects × [5 + 4 + 3 + 3] rows):

- (a) obtained proportions sum to 1 within rounding;
- (b) proportion × 240 is an integer within rounding tolerance.

**First pass (straight from the OCR text layer): 50 / 60 rows passed both checks.**
The 10 failing rows were each re-read from a 360 dpi render of the PDF page.

### Cell corrected (OCR slip, fixed against the image) — 1 cell

| Subject | Matrix | Cell | OCR text layer said | PDF image shows | Effect |
|---|---|---|---|---|---|
| s3 (G.X.) | AEFHX | E → E | `.515` | **`.575`** | row sum .940 → 1.000; counts [21,138,50,14,17] |

### Rows whose flag was a rounding artefact, not a transcription error — 7 rows

Re-reading the image confirmed the OCR text exactly; the flag was caused by the paper's
own third decimal being off by .001. In each case the count vector is still forced
uniquely by the row-sum-240 constraint, so no data is lost:

- s1 AEFHX row E (printed .241 and .091 vs .242 and .092 from counts 58 and 22)
- s1 AEFHX row H (printed .159 vs .158 from count 38)
- s1 AEX row A (printed .620 vs .621 from count 149)
- s2 AEFH row H (printed .091 vs .092 from count 22)
- s2 AEX row A (printed .134 vs .133 from count 32)
- s3 AEFH row A (printed .191 vs .192 from count 46)
- s4 AEFH row A (printed .070 vs .071 from count 17)

### After correction: 58 / 60 rows pass both checks

Re-reading the delivered CSVs: all 58 non-`NA` rows have counts summing to exactly 240,
printed proportions summing to 1 within rounding, and every count reproducing its
printed proportion to within .001. The 2 remaining rows carry one `NA` each.

Across the 207 resolved obtained cells, the printed proportion differs from
`round(count/240, 3)` in 31 cells. 22 of those are exact ties (`count/240` equals
x.xxx5 exactly, which happens whenever count ≡ 3 mod 6) where the direction is purely a
rounding-convention matter. The remaining 9 are genuine ±.001 inconsistencies in the
printed table, listed above; the paper appears to truncate `.xxx1667` downward in most
of them but rounds `.xxx3333` upward twice, so its third decimal should be treated as
±.001 throughout.

### Independent cross-check on the master matrices

The paper's CRR predictions for the subset matrices are, by construction (Eq. 1),
the master-matrix row renormalised over the subset. Recomputing all 136 printed CRR
entries from the recovered master counts reproduces **122 / 136 exactly** (|diff| ≤
.0005), and the other 14 agree to within .0009. Since changing any single master count
by 1 would shift a CRR entry by roughly 1/215 ≈ .005 — an order of magnitude more than
the largest observed discrepancy — this confirms all 100 master-matrix counts
independently of checks (a) and (b).

## Cells NOT resolved — 2 cells, written as `NA`

These are **errors in the published table**, not OCR failures: the image was read at
360 dpi and shows exactly what the text layer reported. The printed row does not sum to
1, so one entry in it is a typesetting error, and the erroneous cell cannot be read from
the PDF. Both are marked `NA` in the counts CSVs. The verbatim printed value is retained
in the corresponding `_prop.csv`.

| Subject | Matrix | Cell | Printed | Printed row | Row sum |
|---|---|---|---|---|---|
| s1 (D.X.) | AEFH | H → A | `.107` | .107 / .113 / .070 / .650 | **0.940** |
| s4 (A.X.) | FHX  | F → H | `.223` | .613 / .223 / .154 | **0.990** |

For each, the arithmetic repair is unique but is an **inference, not a reading**, and is
therefore *not* written into the CSVs. Recorded here so the choice is the user's:

- **s1 AEFH, H → A.** Among all single-decimal-digit substitutions in that row, only
  `.107 → .167` yields a row summing to 1.000 with integer-consistent counts, giving
  count 40 (40/240 = .16667 → .167) and row [40, 27, 17, 156] = 240. Supporting but
  non-decisive evidence: the WSCM prediction for this cell is .196 and the CRR/SSCM
  predictions are .177/.179, and the WSCM tracks the obtained values within about .02
  in every other cell of this matrix — a .107 obtained value would be a .089 miss.
- **s4 FHX, F → H.** Only `.223 → .233` yields a consistent row, giving count 56
  (56/240 = .23333 → .233) and row [147, 56, 37] = 240. Supporting evidence: the WSCM
  prediction for this cell is .234, and the WSCM matches the other two cells of this row
  to within .001 (.613 vs .613, .153 vs .154).

The three other cells in each of those two rows are read directly and pass check (b)
(s1 AEFH row H: E = 27, F = 17, H = 156; s4 FHX row F: F = 147, X = 37), so those
values are retained. Note that with H → A unresolved, s1's F = 17 rests on the printed
`.070` being a truncation of .0708 (the alternative, 16, would print as .067); it is
therefore slightly less certain than the other retained counts.

## Known OCR noise in quadrants NOT digitized

Recorded only so nobody re-derives predictions from the raw text layer. Examples seen:
s3 AEFHX row E SSCM printed `.091` was OCR'd `.91`; s2 AEFHX row E SSCM `.417` was
OCR'd `.411`; s2 AEFH row F CRR `.267` was OCR'd `.261`; s4 AEX row E obtained `.258`
was OCR'd `2'58`. The WSCM / CRR / SSCM quadrants in this release are **not** digitized
at all — only the obtained proportions are.
