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

Two things make it the right first target. The label set is chosen by
the end user and is full of near-synonyms, so IIA fails by construction
rather than by accident. And the label embeddings are already in hand,
so the loadings `V` are a byproduct of the forward pass, not a fit: the
race head is an inference-time swap on a frozen checkpoint. Exactness
costs microseconds at this N. If the swap moves calibration or the
synonym-split behaviour on a public zero-shot suite, that is the
existence proof for the large-N product without paying its kernel bill
first.
