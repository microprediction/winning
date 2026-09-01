
## MoPLEx (Peter, 2026-09-01): PL mixtures for heterogeneous annotators
Li, Zhang, Wang, Zhang, "MoPLEx: Estimating Plackett-Luce Mixture
Models for Multi-Objective Alignment", arXiv:2608.25200, EMNLP 2026
[locator-verified; unread in full]. Mixtures of k Plackett-Luce models
from multi-way rankings by heterogeneous annotators; identifiability
fails for k > m/2 at ranking length m; EM-style estimation with
embedding-space gradients; +43.7% clustering / +15.2% ranking accuracy
over single-PL and BT-mixture baselines on UltraFeedback and PERSONA.

Why it lands here: a PL mixture IS mixed logit for rankings -- the very
family that won the posttraining Luce-vs-probit comparison at finite
budgets. The engine's counter-move is the mixture of correlated
THURSTONE races: exact order likelihoods with exact gradients make the
EM tractable (E-step responsibilities from order_loglik per component,
M-step our calibration), heterogeneity can live in mu OR in the
covariance (annotator cliques as blocks), and the open theory question
worth chasing: their k > m/2 identifiability ceiling is PL-specific --
what is the probit-mixture threshold, and does correlation structure
buy identifiability that independence cannot? Also confounds to
untangle per the bandits exp24 lesson: annotator-mixture effects vs
base-skew misspecification can masquerade as one another.
