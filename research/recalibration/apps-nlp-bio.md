# Application scouting: NLP, speech, bioinformatics

*Front report for the proposed uncertainty layer: frozen scores over N mutually exclusive
labels → latent Gaussian means → low-rank-plus-diagonal covariance (V,D) fit by exact
maximum likelihood on held-out winners → exact correlated probabilities, top-k set
probabilities, and correct restricted-label-set inference (no IIA). Horse racing excluded
by mandate. Scouted 2026-08-25 via web search; quotes are verbatim from the cited source
unless marked (search summary).*

---

## 1. LLM multiple-choice calibration and selective prediction

**Setting.** A frozen LLM scores N answer options (MMLU, ARC, HellaSwag, tool/function
selection). Exactly one option is correct; the per-option logprob is the frozen score.

- **(i) Mutually exclusive winner + frozen scorer?** Yes, cleanly. One gold option per
  question; the scorer is a frozen checkpoint whose per-option logits are reproducible.
- **(ii) Public data/checkpoints.** Abundant: open-weight models (Llama-3, Mistral, Qwen)
  rerun in minutes via lm-evaluation-harness, which records per-option loglikelihoods;
  HELM publishes raw per-instance outputs. MMLU has 14k test questions x 4 options;
  tool-selection benchmarks (ToolBench, BFCL) give varying menus.
- **(iii) Incumbent + verbatim limitation.**
  - Temperature scaling is the default post-hoc fix and is documented as insufficient:
    "When using held out datasets from training distribution... temperature scaling
    significantly deteriorates calibration of the model" and "Post-RLHF models develop
    input-dependent overconfidence... a single temperature can't account for that
    variation" (search summaries of arXiv:2608.07419 and the ATS line,
    aclanthology.org/2024.emnlp-main.1007).
  - Kadavath et al. (arXiv:2207.05221): "larger models are well-calibrated on diverse
    multiple choice and true/false questions when they are provided in the right format"
    — i.e., calibration is fragile to format/menu changes, which is exactly an IIA
    failure surface.
  - Surface form competition (Holtzman et al., EMNLP 2021, arXiv:2104.08315): different
    surface forms "compete for probability mass, even if they represent the same
    underlying concept," lowering the probability of the correct answer (search summary).
    Two synonymous options splitting mass is precisely a positive off-diagonal in V that
    softmax cannot represent.
  - Semantic entropy (Kuhn/Gal/Farquhar arXiv:2302.09664; Farquhar et al., Nature 2024):
    "semantic entropy — an entropy which incorporates linguistic invariances created by
    shared meanings," implemented by hard-clustering answers via bidirectional
    entailment. **This is a crude, binary version of correlated confusability: cluster
    membership = correlation 1, otherwise 0.** A fitted V (e.g., parameterized by option
    embeddings) is the continuous generalization, and it yields exact set probabilities
    for a "semantic cluster" rather than ad hoc entropy over clusters.
- **(iv) Unique advantage.** (a) Top-k / set probabilities: P(answer in {A,C}) computed
  exactly, the object selective prediction actually needs. (b) Restricted-menu: LLM MCQ
  probabilities are known to be inconsistent when distractors are removed or the question
  is re-posed as binary; renormalized softmax assumes IIA, the Gaussian layer does not.
  (c) Cross-menu fitting is natural: every question has a different option set, so V must
  be parameterized in an embedding basis — same trick needed for entity linking below.
- **(v) Minimal demo.** Llama-3-8B per-option logits on MMLU val/test via
  lm-evaluation-harness; fit (V,D) with V spanned by option text embeddings; compare
  ECE / selective-risk-coverage vs temperature scaling and vs semantic-entropy-style
  clustering; ablation: delete one distractor and compare renormalized softmax vs
  restricted Gaussian inference against empirical accuracy on the reduced menus.
- **(vi) Venue.** ACL/EMNLP (main or Findings); UncertaiNLP workshop for a fast first
  strike; TMLR for the methodological version.

**Verdict: strongest overall.** Huge audience, trivially public data, the incumbents'
limitations are documented in their own abstracts, and semantic entropy hands us the
framing: "they discretized the covariance; we fit it."

---

## 2. Speech recognition: confusion networks, confidence, contextual biasing

**Setting.** Confusion networks / N-best lists from a frozen end-to-end ASR model give,
per slot, a small menu of mutually exclusive word hypotheses with scores. Confusion
matrices are indeed ancient here (phoneme confusability, Fisher/IBM-era word confusion
networks), but modern end-to-end systems mostly ship the raw softmax.

- **(i) Mutually exclusive winner + frozen scorer?** Yes at the confusion-network-slot
  level: one reference word per slot, alternatives scored by a frozen Conformer.
- **(ii) Public data/checkpoints.** NVIDIA NeMo pretrained Conformer-CTC and
  Conformer-RNN-T on LibriSpeech; the entropy-confidence paper's code is in NeMo
  (arXiv:2212.08703). Whisper N-best lists are equally accessible.
- **(iii) Incumbent + verbatim limitation.** From arXiv:2212.08703 (verbatim):
  - "The effectiveness of this approach is limited by the so-called prediction
    overconfidence, when the probability distribution is skewed towards the best
    hypothesis."
  - "We consider a model to be overconfident when its median probability of incorrect
    predictions is above 0.9. This situation is typical for overtrained CTC and RNN-T
    models."
  - "To mitigate the overconfidence issue, one can use temperature scaling, dropout,
    ensemble of ASR models etc." — i.e., the field's post-hoc toolkit is temperature-like
    (their Tsallis entropy "works similar to temperature scaling"), all IIA-preserving
    and confusability-blind. Nobody recalibrates over *correlated* word confusions post
    hoc; acoustic similarity (their/there, affect/effect) is exactly an off-diagonal V.
- **(iv) Unique advantage.** (a) Lattice/confusion-network rescoring *is*
  restricted-menu inference: the lattice restricts the label set per slot. (b) Contextual
  biasing is a live, commercially painful restricted-menu problem: bias lists of names/
  drug terms are handled by shallow fusion score hacks with known "overboosting"
  pathologies — "With shallow fusion, an external vocabulary bias is added as a weighted
  score on top of the model's own language model scores" (search summary; see
  arXiv:2012.00133, arXiv:2508.07014). An exact conditional posterior given "the word is
  in this list" replaces a tuned boost weight.
- **(v) Minimal demo.** NeMo Conformer-CTC on LibriSpeech test-other (+noise): build
  confusion-network slots, fit (V,D) on validation winners with V spanned by phonetic
  embeddings; report word-error-detection AUC/NCE vs max-prob, temperature scaling, and
  Tsallis entropy (their published baselines make the comparison table ready-made).
- **(vi) Venue.** Interspeech / ICASSP.

**Verdict: strong.** Best verbatim overconfidence quotes of any front and free baselines,
but the confidence-estimation subfield is crowded with trained estimators; the clean win
is the biasing/restricted-menu angle.

---

## 3. Bioinformatics

### 3a. Cell-type annotation (Cell Ontology labels)

**Setting.** CellTypist (multinomial logistic regression, published frozen models) and
scANVI score each cell against N cell-type labels that are strongly correlated through
the Cell Ontology (CD4 T vs CD8 T vs NKT...).

- **(i)** Yes: one true label per cell at a fixed granularity; CellTypist's frozen models
  emit per-label decision scores/probabilities.
- **(ii)** Public: CellTypist model zoo, Tabula Sapiens, Human Cell Atlas, AzimuthPBMC;
  the hierarchical-reject paper (PMC10957513) ships five public datasets + code on
  GitHub/Zenodo.
- **(iii) Incumbent + verbatim limitation.** Incumbent is a reject option, possibly
  hierarchical. From PMC10957513 (verbatim): "the linear SVM's confidence scores are
  close together for the classes at the top of the hierarchy, which could result in
  different predictions" — scores the authors call "badly calibrated". And:
  "ontology-based hierarchies do not correctly represent transcriptomic relations between
  the cell type labels" — i.e., the hand-built hierarchy used for rejection is the wrong
  correlation structure. Reference-based methods' "outputs often lack calibrated
  uncertainty measures" (search summary, popV/infocusp review).
- **(iv) Unique advantage.** (a) The Cell Ontology tree is a hand-coded, wrong V; ML
  fitting learns the transcriptomic confusability directly, and top-k set probabilities
  give principled "partial rejection" to an ancestor set — exactly what the hierarchical
  reject option approximates with thresholds. (b) Restricted-menu inference is standard
  *practice* already: annotators routinely restrict to tissue-specific label sets
  (CellTypist ships organ-specific models precisely because the full menu misbehaves).
  Conditioning one fitted global model on a tissue's label subset, exactly, is the
  product. Natural cross-menu variation (different tissues = different menus) feeds the
  fit.
- **(v) Minimal demo.** CellTypist Immune_All model on held-out Tabula Sapiens: fit (V,D)
  on winners; show (1) set-probability calibration for ontology siblings, (2) tissue-
  restricted inference beats softmax renormalization of the full model and matches the
  dedicated organ-specific model without retraining.
- **(vi) Venue.** Genome Biology, Bioinformatics, or a Nature Methods brief communication.

### 3b. Taxonomic read classification (Kraken-style)

**Setting.** Kraken2 scores each read against thousands of taxa via k-mer hits;
database subsetting is the everyday operation.

- **(i)** Yes: one true source taxon per (simulated) read; frozen scorer = fixed
  database + k-mer counter.
- **(ii)** Public: Kraken2 + standard/RefSeq databases, CAMI benchmarks, simulated reads
  with known truth.
- **(iii) Incumbent + verbatim limitation.** The incumbent is a single scalar
  "confidence score" (CS) threshold. From PMC11624175 (verbatim): "There are no precise
  guidelines for the choice of CS for Kraken2 and it is mainly determined by the
  researcher's personal choice or the default value." Also: "the inclusion of more
  species can lead to a higher number of false positives, particularly those that are
  closely related to the true species" — correlated confusability among sister taxa,
  documented. And the restricted-menu problem stated outright: "Ideally, the reference
  database should be restricted to the smallest set of domains fully representing taxa
  present in the sample" (search summary of same paper). The RefSeq-growth paper
  (Frontiers 2026) documents that classification *changes as the menu grows*: k-mers
  "previously unique to specific taxa become shared across multiple lineages" (search
  summary) — a pure IIA violation in current practice.
- **(iv) Unique advantage.** Database subsetting = restricted-menu inference is the
  canonical operation of the field, and no tool does it coherently: today you rebuild
  the database and all probabilities move non-probabilistically. A fitted (V,D) over
  taxa (V spanned by taxonomy/ANI embeddings — low rank is *necessary* at N in the
  thousands, which plays to the method) gives per-read posterior taxa probabilities that
  update exactly under database restriction.
- **(v) Minimal demo.** CAMI-style simulation: Kraken2 per-taxon k-mer hit counts as
  scores, fit (V,D) on labeled reads, show calibrated species posteriors and — the
  headline — consistency of abundance estimates under database subsetting vs the
  CS-threshold + Bracken pipeline.
- **(vi) Venue.** Bioinformatics / Genome Biology / ISMB proceedings.
- **Caveat.** Engineering-heaviest option: scores are sparse counts, N is huge, and the
  Gaussian-latent reading of k-mer counts needs care (a monotone link before fitting).

### 3c. Variant pathogenicity (ranked lower)

Mostly binary/5-class with N tiny, so the correlated-menu machinery is underemployed.
The live issue is dependency between evidence *sources*, not labels: "when the
conditional mutual information between sources is non-negligible, directly summing their
calibrated evidence under the ACMG/AMP Bayesian framework may lead to inflated
pathogenicity probabilities" (search summary of P-KNN, bioRxiv 2025.09.24.678417; see
also ClinGen PP3/BP4 calibration, AJHG 2022). Related in spirit (correlated Gaussians
fix double counting) but not our menu structure. Park for now.

---

## 4. Entity linking / retrieval reranking

**Setting.** A frozen bi-encoder (BLINK) retrieves top-K candidate entities per mention;
a cross-encoder scores them; one gold entity (or NIL). Candidate sets differ per mention.

- **(i)** Yes: one gold entity per mention; frozen public scorers.
- **(ii)** Public: facebookresearch/BLINK checkpoints, ZESHEL (zero-shot, varying
  candidate domains), AIDA-CoNLL; BEIR + cross-encoder rerankers for the retrieval twin.
- **(iii) Incumbent + verbatim limitation.** Softmax over candidates plus a NIL
  threshold. Verbatim: "the NIL prediction problem, which aims to identify mentions
  without a corresponding entity in the knowledge base, has received insufficient
  attention" (arXiv:2305.15725). LLM-based EL confidence: models "produce very high
  confidence levels even for datasets where F1 scores are around 90 or lower, with
  confidence histograms also suggesting miscalibration" (search summary,
  arXiv:2510.01251 / 2509.19557). Rerankers: "Encoder-based cross-encoders produce
  poorly calibrated logits" and "the scores are relative, not absolute" (search
  summaries).
- **(iv) Unique advantage.** (a) Per-query candidate sets are exactly the cross-menu
  variation the estimator wants — V must be parameterized by entity embeddings, and EL
  supplies millions of menus. (b) NIL is an outside option: P(gold not in the retrieved
  set) is a top-K *set* probability the layer computes exactly, versus today's tuned
  threshold. (c) Same machinery transfers verbatim to RAG reranking confidence.
- **(v) Minimal demo.** BLINK bi-encoder top-64 on ZESHEL: fit (V,D) with V from entity
  embeddings on validation winners; report NIL detection AUC and calibration of
  "gold-in-top-k" set probabilities vs softmax+threshold.
- **(vi) Venue.** EMNLP/EACL; SIGIR for the reranking variant.

**Verdict: solid, and methodologically important** because it forces the
embedding-parameterized V that fronts 1 and 3b also need — but the miscalibration
complaints here are less crisply documented than in ASR/LLM-MCQ.

---

## Ranking

| Rank | Candidate | Why |
|---|---|---|
| 1 | LLM multiple-choice calibration / selective prediction | Biggest audience; free data; incumbents self-document their limits; semantic entropy is a hard-clustered special case of a fitted V; restricted-menu (option removal) inconsistency is demonstrable in an afternoon |
| 2 | Cell-type annotation with tissue-restricted label sets | Cleanest scores (frozen logistic models), correlated ontology labels are the field's own complaint, restricted menus (organ-specific models) are already standard practice we can subsume |
| 3 | Kraken-style taxonomic classification under database subsetting | Structurally the most perfect restricted-menu story in all of science ("rebuild the database" = change the menu); documented ad hoc incumbent; costliest demo |
| 4 | ASR confusion-network confidence + contextual biasing | Best verbatim overconfidence quotes, NeMo baselines ready; crowded confidence subfield, so lead with the biasing/restricted-menu angle |
| 5 | Entity linking / reranking (NIL as set probability) | Great fitting substrate (menus vary per query), useful NIL story; weaker documented pain |
| 6 | Variant pathogenicity | Real calibration dependency literature but wrong problem shape (N tiny; dependence is across evidence sources, not labels) |

## Sources

- Kadavath et al. 2022, arXiv:2207.05221 — LMs (Mostly) Know What They Know
- Kuhn, Gal, Farquhar 2023, arXiv:2302.09664 — Semantic Uncertainty (and Farquhar et al., Nature 630, 2024)
- Holtzman et al., EMNLP 2021, arXiv:2104.08315 — Surface Form Competition
- Shen et al. (ATS), EMNLP 2024, aclanthology.org/2024.emnlp-main.1007; Bilevel LLM calibration, arXiv:2608.07419
- Laptev et al., arXiv:2212.08703 — Fast Entropy-Based Word-Level Confidence for E2E ASR (NeMo)
- Le et al., arXiv:2012.00133 (unigram shallow fusion); TurboBias, arXiv:2508.07014
- Theunissen et al., PMC10957513 — Uncertainty-aware single-cell annotation with a hierarchical reject option
- popV consensus cell-type prediction, bioRxiv 2023.08.18.553912
- Kraken2 database/confidence study, PMC11624175; RefSeq growth study, Frontiers HPC 2026 (10.3389/fhpcp.2026.1860848)
- P-KNN joint calibration, bioRxiv 2025.09.24.678417; ClinGen PP3/BP4, AJHG 2022 (S0002-9297(22)00461-X)
- Learn to Not Link (NIL), arXiv:2305.15725; BLINK, arXiv:1911.03814; LLM-EL uncertainty, arXiv:2510.01251
