# Application scouting digest — 2026-08-25 overnight run

Four scouts (e-commerce, ML systems, science first-arrival, econ/finance;
racing excluded as Peter's separate workstream) each evaluated candidates
on race-mapping validity, public data, incumbent limitations (verbatim
quotes in the per-front files), unique advantage, and a minimal demo.
Per-front details: `ecommerce.md`, `ml-systems.md`, `science-races.md`,
`econ-finance.md`.

## Cross-front ranking

| # | candidate | front | why it leads | data, today |
|---|---|---|---|---|
| 1 | Heteroskedastic classifier heads | ML | Collier et al. / HET-XL is LITERALLY our model (argmax over N(mu, VV'+D), classes to 21k), approximated by MC + temperature softmax with an admitted bias-variance tau; we delete both approximations with exact p and exact gradients. Drop-in demo against `uncertainty_baselines`. | ImageNet/CIFAR, public |
| 2 | Product-display / assortment choice | e-comm | Expedia impressions randomly sorted (clean menus), JD.com clicks+orders; the EM+EP probit incumbent headline is "more than 100 alternatives" where we do 10^4 exactly. DOUBLE DUTY: this is the real-data cross-menu validation the calibration paper's EDIT-TODO already demands. | Expedia ICDM-13, JD.com MSOM, public |
| 3 | RCV cast-vote records | econ | Full ranking patterns, free, at scale (398+ elections); candidate-exit spoiler analysis IS the deletion ensemble (Alaska 2022, Burlington 2009 as ground truth); nobody has fit correlated Thurstone to CVRs. DOUBLE DUTY: ballots are the bivariate-information data the (V,D)-from-ranks paper needs. | FairVote Dataverse, public |
| 4 | Stockout / delisting substitution | e-comm | The retailer question is verbatim our one-pass removal counterfactual; incumbents run simulation loops around unobserved-choice-set bias; JD.com stockout windows give out-of-sample validation. | JD.com MSOM, public |
| 5 | RLHF K-wise preference modeling | ML | K-wise ranking data (Nectar 7-wise) currently shredded into IIA pairs; BT expressiveness complaints are documented; the correlated-probit likelihood with exact gradient ALREADY EXISTS in this repo. Lowest marginal effort. | Nectar etc., public |
| 6 | DNA replication origin firing | science | Origin efficiencies are win frequencies over millions of cells; incumbent fits by 500-simulations-per-iteration loops and drops spatial correlation; removal counterfactuals map to REAL origin-deletion strains. Highest novelty splash, longest cultural distance. | OriDB, Kronos scRT, public |
| 7 | LLM arena rating | ML | Style confounds + silent deprecation of 205/243 models "can violate key assumptions of the Bradley-Terry model"; removal counterfactuals and max-of-correlated-family are exactly our operations. Partially occupied (pairwise Thurstone-with-covariance exists); our lane is N-way + inversion + counterfactuals. | lmarena 140k battles, public |
| 8 | School choice | econ | Field workhorse is MNP-by-Gibbs, "computationally burdensome", menu-change counterfactuals already benchmarked. Flagship data restricted. | restricted |
| 9 | Multi-alternative cognitive choice | science | "No known explicit solution... 20,000 simulations", N capped at 4; letter-confusion demo trivial. Small-N field culture. | confusion matrices, public |
| 10 | kth-to-default baskets | finance | Structurally our top-k under one-factor copula; citations gathered, NO valuation claims per standing rule; Peter's characterization required. | market data, restricted |
| 11 | Common-cause reliability | science | Alpha-factor single-group restriction is solved by factor loadings; slow-moving field. | fleet data, mostly restricted |

## Suggested immediate moves (pending Peter)

1. Classifier-head demo: exact winner probabilities + gradients vs
   Collier's MC-softmax on a public checkpoint — one experiment file,
   biggest audience per unit work.
2. Expedia/JD.com calibration: serves candidate 2 AND closes the
   calibration paper's "real-data inner-inversion demo" TODO in one
   stroke.
3. RCV ballots: feeds the second paper's identification program with
   real bivariate rank data (small N, many ballots) before the
   marketing-scale version.
