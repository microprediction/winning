# Experiment 39: the heteroskedastic classifier head, exact vs MC-softmax

Collier et al. (CVPR 2021) / HET-XL (ICLR 2023) put the factor Gaussian
race on classifier heads and estimate P(c = argmax) by S=50 Monte Carlo
samples of a temperature softmax. By the Gumbel-argmax identity the
expectation of that estimator at any tau IS a hard race with
tau-Gumbel-convolved noise, which the shared field computes exactly.

Simulated ImageNet-scale head: K=1000, rank 3 (pruned Gauss-Hermite,
1317 nodes), heteroskedastic D, hot correlated classes, one seed.

Anchor: exact tau=0 map vs 20M-draw MC over the 94 classes with >=200
hits: max z 2.45, mean z 0.85, none above 4 - statistically
indistinguishable. MC-mean check confirms the temperature identity at
every tau (max abs deviation matches the 10k-sample rate).

The measured bias-variance dilemma at S=50 (medians by mass stratum;
bias = |dlog| of the tempered probabilities against true argmax
probabilities, i.e. the error remaining at infinite samples):

  tau    sd/p (p>1e-2)  sd/p (1e-4..1e-2)  bias top   bias mid
  1.00   0.163          0.161              0.744      1.137
  0.50   0.358          0.628              0.188      0.745
  0.25   0.474          2.160              0.045      0.242
  0.10   0.542          3.194              0.006      0.031

At tau=1 the head-class probabilities are biased by a factor ~2 no
matter how many samples are drawn; at tau=0.1 the bias is gone but a
mid-mass class carries ~320% relative noise per evaluation. There is no
good tau. The exact map has neither error term: 59s at rank 3 in NumPy
(L=2001, unoptimized), 96s at HET-XL-like rank 15 via Sobol 2^11.

Not yet measured (needed for a paper-grade claim): gradient-variance
comparison (exact JVP vs MC gradient), multiple seeds, and a real
public checkpoint (uncertainty_baselines) instead of a simulated head.
