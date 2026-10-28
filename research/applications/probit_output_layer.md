# The multinomial-probit output layer for large classifiers
(Peter's notes, 2026-09-01. Verdict: highest technical upside; not
the first application to commercialize. All references [U] until
read at source.)

Google's heteroscedastic-classifier line (HET, HET-XL) places a
multivariate Gaussian over class logits with covariance explicitly
parameterized as

    Sigma(x) = V(x)' V(x) + diag d(x),

precisely the factor-probit form, applied at >21,000 classes (HET)
and ~30,000 classes / up to four billion images (HET-XL).

The revealing part is how they evade the Gaussian argmax integral:
they Monte-Carlo E_eps[softmax((mu(x) + eps)/tau)], describe the
softmax as an APPROXIMATION to the underlying discrete-choice
generative process, and report that tau regulates a crucial
bias-variance trade-off with performance sensitive to it; later work
LEARNS the temperature because sweeps are too costly at scale. Here
factor logit is not a competitor -- it is the tractability
compromise the papers introduced after specifying Gaussian latent
logits. The engine computes the specified quantity,

    Pr(mu_i(x) + eps_i(x) = max_j {mu_j(x) + eps_j(x)}),

directly: no Gumbel smoothing, no sampling, no temperature.

## The potential product
A deterministic low-rank multinomial-probit output layer WITH
GRADIENTS. Uses: correlated label noise; semantic confusion among
thousands of classes; uncertainty-calibrated classifiers; retrieval
over enormous label spaces; image-text contrastive classifiers;
medical coding with correlated diagnoses.

## The obstacle
Throughput. A method suitable for an occasional 30,000-alternative
inversion is not automatically suitable for billions of SGD
observations. Needs: a batched GPU/JAX kernel, aggressive class
pruning, ranks ~1-4. Parked behind Tracks A/D/E in PLANS.md; this
note is the fuller statement of why the fit is exact and what the
engineering bill is.

## The small-N front door: zero-shot classifiers (added 2026-09-23)
The throughput obstacle above assumes tens of thousands of classes and
billions of observations. GLiClass (Knowledgator; see ml-systems.md 1b)
is the same head at the other end of the scale: a user-supplied label set
of a few to a few dozen labels, encoded in the same forward pass as the
text, scored by a scalar logit per label, and closed by

    torch.softmax(logits, dim=-1)      # single-label
    torch.sigmoid(logits) + threshold  # multi-label

Two things make it a cheap first test. User-chosen label sets often
contain near-synonyms, where softmax's IIA (conditional on the logits; the
uni-encoder can move logits when labels change) is most likely to bite.
And contextual label embeddings give a candidate covariance kernel
`K = E E^T` from the same forward pass; its rank-r eigenfactor (not `K`
itself) is the loading `V`, with a shared fraction, `D` and the logit
scale left to calibrate, so the swap runs on a frozen checkpoint but is a
hypothesis, not a free byproduct. Cost on the pure-Python path is 1-36 ms
per forward pass at N=3-50, rank 1-2, rising to 0.4-2 s for fields sharp
enough to take the 8,192-node Sobol rule (ml-systems.md 1b,
`gliclass_head_timing.json`). If the calibrated head moves NLL/ECE or the
synonym behaviour on a public zero-shot suite, that is the existence proof
for the large-N product without paying its kernel bill first; if not, the
kernel hypothesis is the first suspect.
