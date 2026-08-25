# Applications scout: industrial scoring and security systems

Front: industrial scoring / security applications for the post-hoc uncertainty layer
(frozen scorer over N mutually exclusive alternatives; scores as latent Gaussian means;
low-rank-plus-diagonal covariance (V,D) fit by ML on held-out outcomes; exact correlated
probabilities, top-k set probabilities, restricted-menu inference).

Excluded by instruction: horse racing/betting. Finance: citations only, no substantive
characterization authored here.

Scouted 2026-08-25. Six candidates evaluated, ranked at the end.

---

## 1. Malware family / intrusion classification

**(i) Mutually exclusive winner + frozen scorer?** Yes. Family attribution is single-label
over hundreds-to-thousands of families (AVClass-style consensus labels). Frozen scorers are
the norm: deployed AV/EDR classifiers are retrained infrequently precisely because
retraining is expensive and audited, while the label distribution drifts continuously —
the exact regime where a cheap post-hoc (V,D) refit on recent labeled outcomes beats
retraining.

**(ii) Public data.** BODMAS (57,293 malware samples, 581 families, timestamped Aug 2019 –
Sep 2020 — built for temporal drift studies; https://whyisyoung.github.io/BODMAS/),
MOTIF (3,095 malware families with expert ground truth, largest public family-labeled
corpus; arXiv:2111.15031), EMBER feature vectors, and the Transcend/Transcendent Android
splits (Drebin/AndroZoo).

**(iii) Incumbent + verbatim limitation.** Incumbent is conformal rejection
(Transcend, USENIX Sec 2017; Transcendent, IEEE S&P 2022). From Transcendent
([arXiv:2010.03856](https://arxiv.org/abs/2010.03856)):

> "Machine learning for malware classification shows encouraging results, but real
> deployments suffer from performance degradation as malware authors adapt their
> techniques to evade detection."

> "One promising method to cope with concept drift is classification with rejection in
> which examples that are likely to be misclassified are instead quarantined until they
> can be expertly analyzed."

The Transcend line explicitly avoids raw classifier confidence because it is unreliable
under drift, and computes per-class p-values instead — i.e., the field has already
conceded that native scores are miscalibrated across families, but its remedy is
*abstention*, not a corrected joint probability over families. Arp et al., "Dos and
Don'ts of Machine Learning in Computer Security" (USENIX Sec 2022,
https://www.usenix.org/conference/usenixsecurity22/presentation/arp) catalog base-rate
misinterpretation (pitfall P8) as endemic in security ML evaluation.

**(iv) Unique advantage.** Shared codebases, packers, and toolkits are literal factors:
families derived from the same source (e.g., Mirai forks, Zeus descendants) have
correlated scores, so a low-rank V has a physical interpretation (loading on shared
code). The layer gives (a) calibrated top-k family sets for analyst triage ("it is one
of these 5 families with 92%"), (b) drift handling by refitting (V,D) monthly on fresh
labels with the scorer frozen, (c) restricted-menu inference when threat intel narrows
the campaign to a subset of families — renormalizing softmax over the subset is wrong
exactly when families are correlated, which is the typical case.

**(v) Minimal demo.** Train LightGBM on BODMAS features (first 6 months), freeze; fit
(V,D) on months 7–8; evaluate NLL/ECE/top-k coverage on months 9–14 vs temperature
scaling and Transcendent-style conformal sets; show restricted-menu inference on the
subset of families active in each evaluation month.

**(vi) Venue/community.** USENIX Security, IEEE S&P, ACSAC, AISec@CCS, DIMVA; industry
side: VirusTotal/Elastic/CrowdStrike ML teams, Camlis conference.

---

## 2. Automatic target recognition / remote-sensing land cover and crop type

**(i) Mutually exclusive winner + frozen scorer?** Yes. Per-pixel/per-parcel single-label
classification (EuroSAT: 10 classes; crop-type maps: one crop per parcel per season).
Frozen scorers are standard practice: foundation models and pretrained CNNs (ResNet on
EuroSAT, Prithvi, SatMAE) are applied across regions and years without retraining.

**(ii) Public data.** Excellent: EuroSAT (27,000 Sentinel-2 chips, 10 classes),
ZueriCrop (48 hierarchical crop classes, 116k parcels), PASTIS, CropHarvest, AAFC
Canadian crop inventory, plus BigEarthNet (multi-label — usable only for the
single-dominant-class variant).

**(iii) Incumbent + verbatim limitation.** Incumbents: raw softmax, temperature scaling,
and recently conformal prediction. Singh et al., "Uncertainty quantification for
probabilistic machine learning in earth observation using conformal prediction"
(Scientific Reports 2024; preprint arXiv:2401.06421) state in their highlights:

> "Data uncertainty is crucial for decision-making. Popular methods lack reliability."

The UQ-EO benchmark paper ([arXiv:2412.06451](https://arxiv.org/abs/2412.06451), "How
Certain are Uncertainty Estimates?") documents that CNN overconfidence is the default
in EO classification. Confusion structure is documented and stable: EuroSAT studies
repeatedly find Annual Crop vs Permanent Crop the dominant confusion (e.g.,
https://www.sciencedirect.com/science/article/pii/S1574954126001925), and crop-type
work attributes it to shared phenology — correlated classes, not label noise.

**(iv) Unique advantage.** (a) Correlation has physical meaning: crops sharing phenology
(winter cereals; grassland/pasture) load on common factors, so rank-2 or rank-3 V is
interpretable and publishable. (b) Restricted-menu inference is the operational reality:
a continental model served in one region should be scored only against crops actually
grown there (regional crop calendars/registers define the legal menu), and independence-style
renormalization over the subset is exactly wrong when the removed classes were correlated
with survivors. No incumbent (temperature scaling, Dirichlet calibration, conformal)
gives coherent subset inference. (c) Top-k set probabilities map to how maps are
consumed ("cereal with 95%" vs a forced single label).

**(v) Minimal demo.** Freeze a published EuroSAT ResNet checkpoint; fit (V,D) on a
held-out country/tile split; report NLL/ECE and top-2 set coverage vs temperature and
Dirichlet scaling; then serve on a region where only 6 of 10 classes occur and compare
restricted-menu inference vs renormalized softmax. Repeat on ZueriCrop with the class
hierarchy as the qualitative check on the fitted factors.

**(vi) Venue/community.** ISPRS Journal, Remote Sensing of Environment, IEEE TGRS,
IGARSS, EarthVision@CVPR; agencies (ESA/Copernicus, USDA NASS CDL users).

---

## 3. Search/ads ranking calibration

**(i) Mutually exclusive winner + frozen scorer?** Partially. Per-impression pCTR is a
Bernoulli per candidate, not one winner among N; the mutually exclusive framing holds
only conditionally ("given exactly one click in the slate, which candidate got it") or
for single-slot auctions. Frozen scorer: yes — industry explicitly separates the ranking
model from a post-hoc calibration layer.

**(ii) Public data.** Criteo and Avazu CTR logs (no slate/session grouping — the main
obstacle for a correlated-candidates demo); Yandex personalized web search
(session-grouped); Open Bandit Dataset (ZOZO) has slate structure.

**(iii) Incumbent + verbatim limitation.** The documented need is loud. MCNet
(Huawei Noah's Ark, WWW 2025, [arXiv:2503.00334](https://arxiv.org/abs/2503.00334)):

> "In online advertising, uncertainty calibration aims to adjust a ranking model's
> probability predictions to better approximate the true likelihood of an event, e.g.,
> a click or a conversion."

> "modern neural networks often struggle to produce accurate probability estimate,
> despite excelling at classification or ranking tasks. This limitation, known as
> miscalibration ... significantly hinder[s] their applications in real-world scenarios."

> "This requires a model to output the predicted CTR that precisely reflects the
> probability of a user clicking on a given advertisement, as it directly influences
> bidding results and, consequently, the platform's revenue."

Related: "Posterior Probability Matters: Doubly-Adaptive Calibration"
(arXiv:2205.07295); Deep Ensemble Shape Calibration (arXiv:2401.09507);
calibration-compatible listwise distillation (arXiv:2312.08727); Instacart engineering
blog on pCTR calibration with transfer learning
(https://tech.instacart.com/calibrating-ctr-prediction-with-transfer-learning-in-instacart-ads-3ec88fa97525).
**Every incumbent calibrates per item (or per field/subset via binning); none models
correlation between candidates competing in the same auction/slate.** Correlation
appears in this literature only at selection time (diversity/submodularity), never in
the calibration layer. That is a genuine gap, but exploiting it requires slate-grouped
outcome data, which is scarce publicly.

**(iv) Unique advantage.** For the "which candidate wins the click/conversion" slice
(single-slot auctions, top-1 recommendation), the layer gives auction-level win
probabilities that respect candidate correlation (same advertiser, same creative
family) and correct inference when the eligible candidate set is filtered at serving
time (budget/targeting filters = restricted menu on every request).

**(v) Minimal demo.** Yandex or Open Bandit slates: freeze a GBDT pCTR model, condition
on one-click sessions, fit (V,D) over candidate-feature factors, compare winner-ID NLL
vs independent calibrated Bernoullis.

**(vi) Venue/community.** KDD (ADS track), WWW, RecSys, AdKDD workshop.

---

## 4. Fraud / AML alert triage (which of N typologies)

**(i) Mutually exclusive winner + frozen scorer?** Weak. Typology labels (structuring,
mule networks, trade-based laundering, ...) are not cleanly mutually exclusive — cases
carry multiple typologies — and the operational decision is suppress/escalate, not
pick-one-of-N. Frozen scorers: yes (model risk management makes retraining slow, and
vendors ship fixed models), which is favorable, but the outcome structure fights the
layer.

**(ii) Public data.** Essentially none real. Synthetic only: IBM AMLSim / AMLworld,
SAML-D, Elliptic (Bitcoin, binary licit/illicit not typologies).

**(iii) Incumbent + verbatim limitation.** Rule engines plus ML triage scoring
(arXiv:2112.07508 "Anti-Money Laundering Alert Optimization Using Machine Learning with
Graphs" describes triage models that suppress/prioritize alerts). Documented pain is
false-positive volume, not typology miscalibration. Flagright
(https://www.flagright.com/post/understanding-false-positives-in-transaction-monitoring):

> "up to 95% of AML alerts are false positives, costing financial institutions millions
> in wasted investigation resources"

> "At an average cost of $500-1,500 per investigation, false positives cost $24-71
> million annually for a single institution."

(Finance-adjacent: citations recorded per instruction; no further characterization.)

**(iv) Unique advantage.** If reframed as "which typology best explains this alert"
for routing to specialist queues, correlated typologies (shared behavioral signatures)
fit the factor structure, and restricted menus arise per jurisdiction/product line.
But the reframing is ours, not the industry's.

**(v) Minimal demo.** SAML-D or AMLworld synthetic typology labels; freeze an XGBoost
typology scorer; fit (V,D); show routing accuracy and calibrated top-2 typology sets.
Demo would be synthetic-only, which limits its persuasive force.

**(vi) Venue/community.** ACM ICAIF (citations-only zone per instruction), KDD applied
track, ACAMS practitioner side.

---

## 5. Industrial fault diagnosis (which of N failure modes)

**(i) Mutually exclusive winner + frozen scorer?** Yes. Benchmarks are single-label:
Tennessee Eastman Process (TEP: 20/21 mutually exclusive programmed faults), CWRU and
Paderborn bearing datasets (fault type x severity classes). Deployed diagnostic models
are frozen between maintenance cycles; recalibration on plant-specific held-out events
is exactly the accepted workflow gap.

**(ii) Public data.** TEP (original + Rieth et al. extended simulation, public),
CWRU bearing, Paderborn, XJTU-SY, MFPT — the most benchmark-saturated field on this
list.

**(iii) Incumbent + verbatim limitation.** Softmax confidence, MC-dropout, evidential
DL, temperature scaling. Calibration complaints are explicit and recent. A 2025
trustworthy-diagnosis architecture paper ([arXiv:2510.03815](https://arxiv.org/pdf/2510.03815))
motivates itself by "core problems of insufficient credibility in industrial fault
diagnosis" and reports "the calibrated ECE was reduced by more than 75%" as a headline
result — i.e., ECE on fault classifiers is a recognized deliverable. The
collaborative human-computer diagnosis literature
(https://www.sciencedirect.com/science/article/abs/pii/S1474034625002423) describes the
problem as models being "overconfident" such that miscalibrated confidence misleads
maintenance personnel into "wrong decisions based on high-confidence incorrect
diagnoses" (safety-critical framing). Bearing transfer studies document "overconfident
misclassification for out of distribution samples" under changed operating conditions.
Per-class confidence is used downstream: maintenance action selection (replace vs
monitor) keys off which fault mode is asserted and how strongly.

**(iv) Unique advantage.** Fault classes are physically correlated through shared
mechanisms (multiple TEP faults propagate through the same reactor/condenser loops;
inner-race vs ball defects share vibration signatures at overlapping frequencies), so
the low-rank factor structure is mechanistically interpretable — a reviewer-friendly
story. Restricted menus are natural: a given machine configuration or operating regime
excludes specific fault modes (a fault in an idle subsystem is off the menu), and
maintenance troubleshooting proceeds by menu narrowing. Top-k set probabilities match
troubleshooting practice (inspect the 3 most probable components).

**(v) Minimal demo.** Freeze a published TEP CNN/GRU classifier; fit (V,D) on one
simulation seed batch; report NLL/ECE and top-3 fault-set coverage vs temperature
scaling; restricted-menu test: score runs where a subset of faults is physically
possible and compare against renormalized softmax. Second demo on CWRU across load
conditions (drift analog).

**(vi) Venue/community.** Mechanical Systems and Signal Processing, Reliability
Engineering & System Safety, IEEE Trans. Industrial Informatics, PHM Society
conference.

---

## 6. Sports/esports outcome models (excluding horse racing)

**(i) Mutually exclusive winner + frozen scorer?** Yes where N > 2: golf tournaments
(~156 entrants, one winner), battle royale (PUBG: ~100 players, one winner), F1,
marathons, tennis draws. Frozen scorer: a skill/strokes-gained model updated weekly is
effectively frozen within an event.

**(ii) Public data.** PGA Tour scoring + OWGR (public), Data Golf publishes predictions
and historical archives; Kaggle PUBG Finish Placement (65k+ matches, ~100 players each,
https://www.kaggle.com/c/pubg-finish-placement-prediction/); esports APIs (OpenDota,
Riot).

**(iii) Incumbent + verbatim limitation.** Data Golf's methodology
(https://datagolf.com/predictive-model-methodology/) is the clearest incumbent: golfer
performance "modeled as normally distributed with some unknown mean and variance" and
"the probability of any tournament result can be estimated through simulation" — i.e.,
*independent* Gaussians pushed through Monte Carlo. Correlation between entrants
(weather waves, course-fit clusters) is not in the covariance, and subset inference
after withdrawals is redone by re-simulation. On the esports side, the
confidence-calibrated MOBA predictor ([arXiv:2006.15521](https://arxiv.org/abs/2006.15521))
documents miscalibration in win predictors: "We propose a novel calibration method that
takes data uncertainty into consideration," reporting ECE improvement from 1.11%
(temperature scaling) to 0.57% — but that is binary (2 teams), not N-entrant.

**(iv) Unique advantage.** This is mechanically the home game (N latent Gaussians, one
winner) minus the excluded domain: exact correlated winner/top-k probabilities replace
Monte Carlo; made-cut / top-20 are top-k set probabilities computed exactly;
withdrawals and alternates are literal restricted-menu inference at serving time.
Correlation factors (tee-time wave weather, course-fit) are estimable from public
outcomes.

**(v) Minimal demo.** Freeze a simple strokes-gained mean model from public PGA data;
fit (V,D) with a wave/weather factor on two seasons of outcomes; compare winner and
top-20 NLL vs the independent-simulation baseline; show exact re-inference after
withdrawals. Or: PUBG Kaggle, per-match winner among ~100 players from pre-final
stats.

**(vi) Venue/community.** Journal of Quantitative Analysis in Sports, MIT Sloan Sports
Analytics Conference, MathSport International; esports: AIIDE, IEEE CoG.

---

## Ranking

1. **Remote sensing / crop type** — perfect mutual exclusivity, the best public data
   (EuroSAT/ZueriCrop/PASTIS), documented stable confusion structure driven by shared
   phenology (interpretable factors), and restricted-menu inference is the *operational
   default* (region-specific crop lists). Incumbents (temperature scaling, conformal)
   have no coherent subset story. Fast, clean demo; receptive venues.
2. **Industrial fault diagnosis** — mutually exclusive fault classes on saturated public
   benchmarks (TEP, CWRU), explicit published complaints about overconfident diagnosis
   confidence driving wrong maintenance decisions, physically interpretable factor
   structure, natural top-k troubleshooting sets and regime-based restricted menus.
3. **Malware family classification** — strongest factor story (shared codebases) and a
   drift-recalibration narrative the field already accepts (Transcend line), with
   timestamped public data (BODMAS/MOTIF); ranked third only because the incumbent
   paradigm is rejection rather than probability, so the pitch requires reframing, and
   security venues demand heavier systems evaluation.
4. **Sports (golf / battle royale)** — mechanically the closest match (independent-
   Gaussian simulation is the named incumbent; withdrawals = restricted menu; made-cut
   = top-k), held back by adjacency to the excluded domain and thinner academic venues.
5. **Ads/search calibration** — loudest documented industry need and a genuine gap
   (no incumbent calibrates across correlated candidates), but the one-winner framing
   is conditional and public slate-grouped data is scarce.
6. **Fraud/AML triage** — frozen scorers and correlated typologies exist, but typologies
   are not mutually exclusive in practice and there is no credible public data; demo
   would be synthetic-only. (Finance-adjacent: citations only.)

## Source list

- Transcendent: Barbero et al., IEEE S&P 2022, https://arxiv.org/abs/2010.03856
- Jordaney et al., Transcend, USENIX Security 2017, https://www.usenix.org/system/files/conference/usenixsecurity17/sec17-jordaney.pdf
- Arp et al., Dos and Don'ts of ML in Computer Security, USENIX Security 2022, https://www.usenix.org/conference/usenixsecurity22/presentation/arp
- BODMAS dataset: https://whyisyoung.github.io/BODMAS/ ; MOTIF: arXiv:2111.15031
- Singh et al., conformal prediction for EO, Sci. Reports 2024, arXiv:2401.06421
- UQ-EO benchmarks, arXiv:2412.06451
- EuroSAT confusion analyses, https://www.sciencedirect.com/science/article/pii/S1574954126001925
- MCNet, WWW 2025, arXiv:2503.00334; Posterior Probability Matters, arXiv:2205.07295;
  DESC, arXiv:2401.09507; listwise distillation, arXiv:2312.08727
- Instacart pCTR calibration blog, https://tech.instacart.com/calibrating-ctr-prediction-with-transfer-learning-in-instacart-ads-3ec88fa97525
- AML alert optimization, arXiv:2112.07508; Flagright FP statistics, https://www.flagright.com/post/understanding-false-positives-in-transaction-monitoring
- Trustworthy fault diagnosis architecture, arXiv:2510.03815; uncertainty-informed
  framework, https://www.sciencedirect.com/science/article/abs/pii/S0951832022004823 ;
  calibrated confidence for human-computer diagnosis, https://www.sciencedirect.com/science/article/abs/pii/S1474034625002423
- Data Golf methodology, https://datagolf.com/predictive-model-methodology/
- Confidence-calibrated MOBA predictor, arXiv:2006.15521
- PUBG Kaggle competition, https://www.kaggle.com/c/pubg-finish-placement-prediction/
