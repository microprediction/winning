"""winning.torch: Thurstone heads for PyTorch (optional; needs torch).

    race_log_prob(mu, sd, base=, points=, mask=)     log P(i wins), batched,
                                                    analytic O(n L) backward
    race_log_prob_factor(mu, V, sd, nodes=)          rank-1/2 factor race,
                                                    Gauss-Hermite node mixture
    thurstone_nll(mu, winner, ...)                   -log P(observed winner)
    ordered_prefix_log_prob(mu, order, ...)          log P(order[0] first, ...)
    topk_log_prob(mu, k, ...)                        log P(i among the k best)

Min-wins (lower mu is better), as in the rest of winning: a network that
emits utilities passes mu = -utility. Importing winning never imports
torch; only importing this subpackage does. Design notes, the lattice
rule, accuracy and cost: docs/design/torch.md.
"""
try:
    import torch  # noqa: F401
except ModuleNotFoundError as exc:          # pragma: no cover
    raise ModuleNotFoundError(
        "No module named 'torch': winning.torch needs PyTorch "
        "(pip install 'winning[torch]')", name="torch") from exc

from .race import race_log_prob, race_lattice
from .factor import race_log_prob_factor
from .losses import thurstone_nll, ordered_prefix_log_prob, topk_log_prob

__all__ = ["race_log_prob", "race_lattice", "race_log_prob_factor",
           "thurstone_nll", "ordered_prefix_log_prob", "topk_log_prob"]
