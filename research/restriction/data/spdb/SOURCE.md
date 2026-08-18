# UIUC Speech Perception Database: response menus of 17 vs 73 on identical tokens

Feipeng Li, Beckman Institute, University of Illinois, 2007-2009. No DOI, no landing page,
no README, no licence. Sole source: https://jontalle.web.engr.illinois.edu/Public/HSR/SPDB.tgz
(88,670,135 bytes), mirrored at http://jontallen.ece.illinois.edu/Public/HSR/SPDB.tgz

MIRRORED HERE BECAUSE IT EXISTS NOWHERE ELSE. No archival copy, no DOI, one sysadmin
decision from vanishing. Only DB/*.mat, the per-trial tables, are kept; the full tarball also
carries stimulus wavs and analysis code. A companion archive of raw logs and the 896 stimulus
wavs is at https://jontalle.web.engr.illinois.edu/uploads/MillerNicely_Vol1.tgz (191 MB) and
is NOT mirrored here.

## Structure

MATLAB v5, one row per trial, 24 columns: exp, utter, talker, listener, hearing, audiogram,
L1, accent, gender, age, stim, compensate, trunctime, frequency, snr, noise, s_level,
n_level, r_level, resp, resptime, repeat, hit, comment. Read with scipy.io.loadmat. The
'utter' field is the individual wav filename, so a response resolves to a specific physical
token.

Menu sizes by experiment: CV06SWN 73 (116,494 trials), MN64 65 (101,760), CV06WN 73 (67,862),
MN16R 17 (64,153), SL05 16, TR06 25, HL07 17, TR07 18, SL07 17, SL06 16.

## Why it is usable, and the confound

The comparisons worth running, all on IDENTICAL wav tokens:
  MN16R (17 alternatives) vs CV06WN (73), white noise, 126 shared tokens, 9 shared
  alternatives, shared SNRs -15,-12,-6,0,+6
  MN64 (65) vs CV06SWN (73), speech-weighted noise, 323 shared tokens, 28 shared alternatives
  SL06 (16) vs SL07 (17), strictly nested, adds only the "?" option

These are overlapping rather than strictly nested for the large pairs: MN64 is 16 consonants
x 4 vowels, CV06 is 9 x 8. Strict nesting exists only where the menu changes by one item.

THE CONFOUND, stated rather than buried: listener overlap across experiments is exactly zero.
The menu manipulation is BETWEEN subjects, so it is confounded with listener sample. Talker
and wav token can be matched; listener cannot.

## Reported result, not yet reproduced here

The agent that found this ran the IIA test SNR-stratified on MN16R vs CV06WN and reports
failure at every SNR: chi2 222.6 (df 72) at -15 dB, 335.8 at -12, 266.3 at -6, 134.6 at 0,
109.0 at +6, grand chi2 1068.3 on df 288, p about 4e-90. Accuracy moves in both directions,
e.g. one token from 0.518 to 0.117 and another from 0.733 to 0.784. Our own two-map
comparison has NOT been run.

Symbol key: https://jontalle.web.engr.illinois.edu/Public/Corpus/LDC_symbols.pdf
xs=esh, xt=theta, xd=eth, xz=ezh, xc=tsh, xj=dzh, xg=eng, xq=ash, xi=small-cap-i
