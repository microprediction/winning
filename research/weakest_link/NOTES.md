# Track C: weakest-link, identifiable zone noise
(Adjudicated PURSUE at one-experiment scale 2026-09-01; report in
../adjudications/weakest_link.md.)

## The claim under attack, in their words (to verify before quoting)
He & Wong (arXiv:2608.01261) fix the Softmin sharpness k, claiming
it non-identifiable. The counter-result: with zone-level noise
e_ij ~ N(0, sigma_e^2) in log-strength, the failure zone is an exact
5-horse Gaussian race and sigma_e IS identifiable (the observed
failure load pins the scale). Their argument holds only at zero zone
noise. NOT the pitch: exactness alone (their independent n=5 joint
likelihood is five lines of Stan).

## Decisive experiment
Refit on the public crossarm data (198 specimens; OSU thesis
zc77sw48x -- bot-gated, machine-readability UNVERIFIED, this is the
data kill test): (i) posterior for sigma_e, (ii) held-out zone log
loss / Brier vs their k=5, (iii) stress-profile gamma distortion vs
k. Then a knot-cluster spatial factor. Authors contacted only after
(i)-(ii) succeed.

## Status
Not started. First action when opened: obtain the thesis data by
hand in a browser and check it is machine-readable.
