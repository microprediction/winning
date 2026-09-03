# Quantization as order statistics on a lattice
(Peter's pointer, 2026-09-03: think about quantization. The folk
explainer -- 95% of weights near zero, outliers carry the quality,
overparameterization absorbs rounding noise -- maps onto this
repository's machinery in three specific places. All literature
references [U] until read.)

## 1. The quantizer IS a lattice-atom representation
NF4 spaces its sixteen levels non-uniformly to match the weight
distribution -- the Lloyd-Max/companding idea: choose atoms to
minimize expected distortion under the density. The winning engine's
native representation is exactly a density on a lattice of atoms,
with the multiplicity calculus handling ties and point masses. Any
question of the form "what does discretizing this distribution to L
atoms do to a downstream max/argmax/race quantity" is a question the
engine answers exactly rather than by simulation: quantized-weight
races, argmax stability under rounding, tie inflation at coarse bit
widths.

## 2. Per-block scales are extreme order statistics; the range law
## is exact in the grammar
Group/block quantization (group sizes 32-128) sets each block's
scale from the block's max |w|; absmax scaling error is driven by
the block RANGE. Under a factor-correlated Gaussian (or any grammar)
model of weights within and across blocks, the distributions of
per-block max, range, and the resulting quantization SNR are exactly
the max/range laws of research/orderstats/SPACINGS.md -- measured
there to n = 2e4 with the shifted and interval survival fields. That
yields ANALYTIC error laws for group quantization: how SNR varies
with group size, correlation, and tail weight, where today the
choices (64 vs 128, per-channel vs per-group) are tuned empirically.
Correlated weights (rows sharing input features = factor loadings)
are the realistic case and exactly what the grammar covers.

## 3. Outlier protection is tail attribution
GPTQ/AWQ protect the <1% outlier weights chosen by Hessian or
activation heuristics. The duplicates-versus-specialist result
(cavity_calculus exp1_shapley) is the same phenomenon one level up:
the components that matter for an extremal objective are the ones
with irreplaceable tail contribution, and deletion value or Shapley
value -- computable for E-max-type objectives from one shared field
-- is the principled version of "which weights are critical." A
weight-group's deletion value under a quality objective is the
attribution GPTQ approximates with curvature. Whether this is
computationally competitive at model scale is untested; as a
diagnostic on small models it is a runnable experiment.

## Cheapest first probe
Model a weight block as factor-Gaussian with a tail-heavy base
(student_base ships), compute the exact distribution of per-block
absmax scale and quantization MSE across group sizes, and check the
analytic law against empirical quantization of a real checkpoint's
layers (a few MB of weights, no GPU). If the law predicts the
per-layer SNR profile, the note writes itself: closed-form group-size
selection replacing the folklore constants.
