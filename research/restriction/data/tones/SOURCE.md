# Absolute identification of tones, nested stimulus sets

Stewart, N., Brown, G. D. A., & Chater, N. (2005). Absolute identification by relative
judgment. Psychological Review, 112(4), 881-911. doi 10.1037/0033-295X.112.4.881

Confusion matrices digitized from Figure 17 of the open-access author manuscript,
https://wrap.warwick.ac.uk/622/1/WRAP_Stewart_absolute_identification.pdf page 104, whose
figures are true vector graphics, so curve vertices give exact values rather than pixel
estimates.

## Why these are usable

Verbatim from the Method: "In the set-size-8 conditions, only the middle 8 tones were used.
Similarly, in the set-size-6 conditions, only the middle 6 tones were used." So S6 inside S8
inside S10 over physically identical tones, crossed with narrow (6 per cent frequency steps)
and wide (12 per cent) spacing. Confirmed independently by Brown, Marley, Donkin and
Heathcote (2008, p. 420).

Rows are stimuli, columns responses, entries P(response | stimulus), participant-averaged.
The restriction test is direct: take an N=10 row for a stimulus in the middle six, restrict
its response distribution to the middle-six responses, and predict the corresponding N=6 row.
Six rows per comparison, two spacing conditions, same again at N=8.

## Provenance and caveats

Validated three ways: every row sums to 1.000; peaks match an independent raster rendering in
Brown et al. (2008) Figure 18; and averaging narrow and wide reproduces the paper's own
spacing-collapsed Figure 21 to three decimals, e.g. N=6 stimulus 1 gives (0.747+0.864)/2 =
0.806 against 0.805 printed.

Precision about +/- 0.002. Values are averaged over 20 participants per cell, pooled over two
response-key mappings, blocks 3 to 7, first 10 trials of each block dropped. Design is
BETWEEN subjects on both set size and spacing, 120 undergraduates, 840 trials each, so no
clustering is possible.

One row is interpolated rather than read: N=10 stimulus 10, where the PDF writer dropped two
collinear near-zero vertices. The other 56 rows are direct vertex reads.

## Raw trial data

Once public at stewart.psych.warwick.ac.uk/publications/RJM_reply/ as
Stewart_Brown_Chater_2005_data.zip, 380 KB, dated 13 Nov 2006. Wayback preserved the directory
listing but not the files, and no OSF, figshare, Zenodo, Dryad or GitHub copy exists.
Recoverable only by asking Neil Stewart (neil.stewart@wbs.ac.uk) for that exact filename.

## Related nested work, no public data

Lacouture & Marley (1995), sets of 2, 4, 6, 8, 10 with smaller sets in the middle of the
larger, accuracy and d-prime only. Kent & Lamberts (2005), the within-subject nested design,
only 3 participants. Both shared privately for re-analyses, never deposited.
