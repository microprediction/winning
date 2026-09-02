# Track E: P(best) on shared evals (flagship demo)
(Adjudicated PURSUE 2026-09-01; report in
../adjudications/model_selection.md.)

## Pitch discipline
Question-bootstrap DOES capture cross-model correlation. The wedges
are: removal counterfactuals (no incumbent), small-P(best) tails
(B=1000 makes P=0.003 three counts), correlation-aware sequential
design vs MODEL SELECTOR's naive-Bayes independence (their
conditional-independence assumption must be quoted from
arXiv:2410.13609 and read in full before use).

## Decisive demo
One two-panel figure on MODEL SELECTOR's own matrices (fallback:
HELM GCS bucket crfm-helm-public per_instance_stats.json): (left)
exact P(best) vs their independence posterior vs question-bootstrap
at matched budget; (right) delete the top model, survivor P(best)
under Luce renormalization vs exact re-pricing. Skip Chatbot Arena
(pairwise battles, no N x R matrix).

## Status
Not started. First action when opened: clone
github.com/RobustML-Lab/model-selector and check the on-disk format
of resources/datasets (UNVERIFIED).
