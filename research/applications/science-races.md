# Science races: application scouting for exact correlated-argmax probabilities

Scouting date: 2026-08-25. Front: natural sciences and decision science, where "which of N wins first"
is the observable. Capability being placed: for N alternatives with latent Gaussian scores sharing a few
common factors (cov = VV' + diag(D)), we compute all N win/first-arrival probabilities exactly in
O(QNL), invert observed win-frequencies to latent abilities, and price removal counterfactuals in one
shared pass. Top-k inclusion in progress. Horse racing excluded by mandate.

Quote hygiene: quotes marked [fetched] were extracted from the article page by direct fetch; quotes
marked [snippet] came via search-result summaries and must be re-verified against the source before
being used in print.

---

## Ranking

| Rank | Candidate | Observable = win freq? | Data | Documented pain | Fit to capability |
|---|---|---|---|---|---|
| 1 | DNA replication origin firing | Yes (origin efficiency across cells) | Public, rich | Simulation-only fitting; correlations dropped | Excellent, incl. removal counterfactual |
| 2 | Multi-alternative choice & identification (cog. sci.) | Yes (choice/confusion frequencies) | Public, huge | 20,000-sim likelihoods; N capped at ~4 | Excellent for choice frequencies; RT margins partial |
| 3 | First-failure-cause attribution with common-cause factors | Yes (failure-cause counts) | Semi-public | Alpha-factor: one CCF group per component | Strong; factor loadings = coupling mechanisms |
| 4 | Clonal interference / lineage barcoding | Partly (which lineage establishes) | Public (Levy/LTEE) | Simulation-heavy inference | Moderate; dynamics not static Gaussian |
| 5 | Immunodominance (T/B-cell clone competition) | Partly (dominance hierarchies) | Public repertoires | Multi-scale simulation models | Moderate |
| 6 | Weakest-link fracture / nucleation sites | Rarely recorded as location freq | Sparse | Weibull misfits documented; dependence rarely modeled | Conceptually clean, data-poor |

---

## 1. DNA replication origin firing competition (genetics) — RANK 1

**The race.** In each S phase, hundreds (yeast) to tens of thousands (human) of licensed replication
origins race to fire; an origin that fires late enough is passively replicated by a fork from a
neighbor and never fires. "Origin efficiency" — the fraction of cells in which a given origin fires
rather than being passively replicated — is a per-alternative win frequency measured across millions of
cells. This is literally a first-arrival race with elimination.

**(i) Win frequencies?** Yes. Origin efficiency and firing-time distributions are the primary
observables of the field (ORM, MATAC-seq, single-cell Repli-seq). MATAC-seq authors: origin firing is
stochastic per cell and "the molecular basis for heterogeneity in efficiency and timing of individual
origins is a long-standing question" [snippet, NAR 2023, gkad1022].

**(ii) Data.** Excellent and public: OriDB (yeast origins), Müller et al. 2014 yeast timing profiles,
Kronos scRT single-cell replication timing (Nat Commun 2022, data on GEO), optical replication mapping
of early-firing human origins (bioRxiv 214841), MATAC-seq single-molecule data.

**(iii) Incumbent + complaint.** State of the art is stochastic simulation fitted iteratively. The 2025
PLoS Comput Biol yeast model (journals.plos.org/ploscompbiol, 10.1371/journal.pcbi.1013066) models 626
origins ("'Confirmed' origins (410) and 'Likely' origins (216)" [fetched]) where "origins compete to
associate with limited firing factors, needed for activation, which then recycle to be used again"
[fetched]. Fitting: "500 simulations of each model" per cycle, "fit for 15 iterations" [fetched] — a
simulation-in-the-loop estimator with no exact likelihood. And correlation is explicitly dropped: "by
not accounting for differences in the time taken for firing factors to diffuse to different origins,
our model does not incorporate the effects of spatial proximity" [fetched]. Meanwhile the biology says
firing propensities ARE correlated — chromatin state, subnuclear compartments, limiting-factor
competition (Genome Research 25:1886: MCM load; PLoS Genetics 2019: initiation-factor levels shift
efficiencies globally).

**(iv) Our advantage.** Replace simulation-in-the-loop with an exact map: latent Gaussian firing
propensity per origin, factors = chromatin/compartment/limiting-factor axes, win probabilities =
efficiencies computed exactly for all N in one pass; invert observed efficiencies to latent origin
strengths. The removal counterfactual is not hypothetical here — origin-deletion strains are a
standard experiment (chromosome III with multiple origins deleted "replicate[s] relatively normally"
[snippet]; neighbors' efficiencies rise to compensate). We can predict the full redistribution map for
every possible deletion in one shared pass and validate against published deletion strains.

**(v) Minimal demo.** Yeast: take Müller et al. efficiencies + timing for the 626 OriDB origins, fit
factor-Gaussian race (factors: chromosome arm, centromere/telomere proximity, Rpd3/Fkh chromatin
class), compare fit and wall-clock against the PLoS Comput Biol simulation pipeline; then predict the
known origin-deletion strains out of sample. Local races (windows between strong origins) keep the
passive-replication geometry honest.

**(vi) Venue.** PLoS Computational Biology, Genome Research, NAR; communities around replication
timing (Gilbert, Nieduszynski, Rhind labs). Rhind's "f = m*a" framework review (PMC8872135) is the
conceptual anchor to cite.

**Caveat.** The genome is 1-D: passive replication makes the race spatially structured, not a pure
argmax among N. The honest framing is nested local races (which origin in a window fires first), which
is exactly an argmax; genome-wide it becomes an argmax with a known adjacency filter.

---

## 2. Multi-alternative choice, absolute identification, confusion matrices (cognitive science) — RANK 2

**(i) Win frequencies?** Yes. Choice proportions across repeated trials are the core observable;
race models (LBA, racing diffusion) add RT. Absolute identification and letter identification yield
full N×N confusion matrices (e.g., a "26 × 26 confusion matrix based on 3,900 trials" in the classic
alphabet studies, Attention Perception & Psychophysics).

**(ii) Data.** Vast and public: OSF archives of choice-RT experiments, absolute-identification data
(Brown/Donkin), classic confusion matrices (Townsend 1971 and successors), best-worst scaling datasets
(Marley, Islam & Hawkins 2016), megastudy repositories.

**(iii) Incumbent + complaint.** The workhorses are INDEPENDENT race models. Brown & Heathcote's LBA
paper itself notes independence is the odd one out: "linear and independent evidence accumulation is a
rare assumption amongst models of choice RT, which usually include response competition explicitly"
[snippet, Brown & Heathcote 2008, Cognitive Psychology]. Heathcote & Matzke's 2022 review states the
tractability bind directly: "Most of the race models discussed here have been analytically tractable
enough to yield an easily computed likelihood, the key quantity required for fitting the models to
data in a comprehensive way. However, requiring such tractability can limit the scope of potential
applications." [fetched, Curr Dir Psychol Sci, 10.1177/09637214221095852]. When correlation IS wanted,
practitioners fall back to brute force: Leite & Ratcliff, fitting multi-alternative models with
"negatively correlated" starting points, report "Because there is no known explicit solution,
predictions from the models were obtained by simulation. We used Monte Carlo methods that generated
20,000 simulations of the decision process" [fetched, PMC2805113] — and their N tops out at 4
alternatives. Thurstonian identification models are prized precisely because "they account for
non-independence of alternatives" [snippet], but full-covariance Thurstone fitting lives at N≈10 via
MCMC (Bayesian Thurstonian models in JAGS, Behav Res Methods 2012). Nobody fits a correlated race to a
26-letter confusion matrix, let alone lexical-scale N.

**(iv) Our advantage.** Exact choice probabilities for correlated Gaussian racers at N in the
thousands: (a) Thurstonian confusion-matrix models where factor loadings ARE the similarity structure
(letters sharing strokes load on shared factors — the confusion matrix is the win-frequency matrix
conditioned on the presented stimulus); (b) absolute identification with a low-rank "anchor" factor
structure; (c) choice-only margins of correlated LBA-type models without simulation. Inversion gives
latent discriminability per item; the removal pass prices context/set-size effects (Hick's law
manipulations, distractor removal) exactly — set-composition effects are a live topic (MLBA, context
effects).

**(v) Minimal demo.** Fit a Q-factor Thurstonian model to a public 26×26 alphabetic confusion matrix,
beat the Luce biased-choice model and independent-Thurstone baselines on held-out cells, and show
exact set-removal predictions (identification among a reduced alphabet) against published reduced-set
conditions. Zero simulation, seconds of compute; then scale the same code to a 1,000-word auditory
confusion demo to show the O(QNL) headroom.

**(vi) Venue.** Psychonomic Bulletin & Review, JMP, Behavior Research Methods; Society for Mathematical
Psychology (mathpsych.org runs an "Evidence Accumulation: Race Models" session every year).

**Caveat.** The RT-modeling community will ask for full choice+RT likelihoods, not just win
probabilities; first-arrival probabilities cover the choice margin and arrival-order, not the full RT
density. Position the tool as the choice-margin/confusion-matrix engine and the exact-N workhorse the
simulation methods calibrate against.

---

## 3. First-failure-cause attribution with common-cause factors (reliability) — RANK 3

**(i) Win frequencies?** Yes. Competing-risks data record which cause/component failed first, across a
fleet of repeated "races" (systems). Cause-of-failure counts across many units are exactly win
frequencies.

**(ii) Data.** NRC/INL common-cause failure database (NUREG/CR-5485 lineage; partly public via
nrcoe.inl.gov), turbofan/bearing degradation sets (NASA), medical-device and drive-fleet failure-mode
data (Backblaze publishes drive failure by model — cause attribution is coarser).

**(iii) Incumbent + complaint.** PRA practice uses the alpha-factor parameterization (NRC SPAR
models). Documented structural limit, addressed in a 2026 Reliability Engineering & System Safety
paper ("An extension to the Alpha Factor method for enhanced common cause failure analysis",
S0951832025011640): "One of the main limitations of the existing Alpha Factor method is that each
component belongs only to a single common cause failure group" [snippet] — i.e., current practice
cannot express a component coupled to several mechanisms at once. Statistical competing-risks work
concedes dependence matters but retreats to bivariate copulas (Marshall-Olkin) or frailties;
latent-failure-time dependence is classically non-identifiable without structure (Tsiatis 1975).

**(iv) Our advantage.** A factor covariance IS multi-group membership: component i loads on factor q
with weight V_iq (coupling mechanisms: shared coolant, shared manufacturer, shared location), and the
exact all-N map gives every component's probability of being the first failure, plus the one-pass
removal counterfactual — "if we harden/remove component i, where does first-failure probability go" —
which is precisely the design question PRA answers today by re-running fault trees. The factor
structure also resolves the identifiability impasse: low-rank dependence + observed first-failure
frequencies across operating conditions is estimable.

**(v) Minimal demo.** Simulate-or-reanalyze a published CCF dataset (NUREG alpha-factor worked
examples): show that a 2-factor Gaussian latent model reproduces alpha factors when groups are
disjoint, and beats them when a component belongs to two coupling mechanisms; report the full removal
map for a 100-component system in milliseconds.

**(vi) Venue.** Reliability Engineering & System Safety, PSAM conference, IEEE Trans. Reliability.

**Caveat.** Nuclear PRA is conservative and validation-bound; the faster wins may be data-center /
fleet-maintenance analytics where the same math has no regulator.

---

## 4. Clonal interference: which mutation/lineage fixes first (evolution) — RANK 4

**(i)** Partly. Fixation is a single race per population, so "win frequencies" require many replicate
populations (Lenski LTEE: 12; barcode lineage tracking: one race but ~500k simultaneously competing
lineages whose establishment probabilities act like win probabilities). Gerrish-Lenski theory is
explicitly a race among beneficial mutations, and its known failure is dependence-adjacent: the theory
"neglects the occurrence of multiple mutations, assuming all mutations arise from the wild-type"
[snippet], which "breaks down in the clonal interference regime where Nμ ≫ 1".

**(ii)** Data public and superb: LTEE metagenomics (Good et al. 2017), Levy et al. 2015 barcode
tracking.

**(iii)** Incumbent: branching-process and traveling-wave theory plus heavy simulation; inference of
per-lineage fitness from trajectories (FitSeq) exists and works, which weakens the pain point.

**(iv/v)** Our angle: treat establishment (which barcode lineages reach threshold first) as a
correlated race — lineages sharing a mutational class share a factor; invert win frequencies to
fitness classes and price "remove the top clone" counterfactuals (relevant to evolutionary rescue and
resistance management). Demo on Levy et al. barcode counts.

**(vi)** Genetics, eLife, PLoS Biology; SMBE.

**Caveat.** The race is dynamic (frequency-dependent, serial dilution), not a static Gaussian argmax;
the mapping is more metaphorical than in ranks 1–3. Rank it as exploratory.

---

## 5. Immunodominance: which clone wins the response (immunology) — RANK 5

**(i)** Dominance hierarchies among epitopes/clones are repeatably measured (cross-competition
experiments; RNA-vaccination hierarchy papers on bioRxiv 2025), and competition is the accepted
mechanism: dominant clones "outcompete and eventually outnumber sub-dominant T cell responses"
[snippet, Eur J Immunol 1999]. Precursor frequency + affinity + shared APC access = a factor structure.
**(ii)** Repertoire sequencing data are public (immuneACCESS, iReceptor).
**(iii)** Incumbents are multi-scale ODE/quasispecies simulations (e.g., arXiv:2310.10966 germinal-center
model) — no exact likelihood, no inversion from observed dominance frequencies.
**(iv/v)** Our angle: invert observed epitope dominance frequencies across many hosts to latent clone
strengths with an APC-access factor; removal counterfactual = epitope deletion/mutation (immune escape
prediction: if this epitope is lost, where does dominance go — directly relevant to vaccine design).
Demo: published hierarchy shifts after epitope knockout in LCMV/flu mouse data.
**(vi)** PNAS, eLife, J Immunol theory sections.
**Caveat.** Same dynamic-race objection as clonal interference; hosts differ in MHC (covariates
needed). High ceiling (vaccine design), higher modeling risk.

---

## 6. Weakest-link fracture and nucleation-site competition (materials) — RANK 6

**(i)** The Weibull weakest-link model is the independent-minimum analog (Weibull ↔ min-stable as
Gumbel ↔ max-stable), and the misfit of independence is documented: "There exists mounting
experimental evidence showing that in some cases the Weibull distribution fails to fit data related to
failure locally initiated by flaws" [snippet]; with strong/correlated disorder "neither Weibull nor
Gumbel distribution applies well" [snippet, arXiv:0901.3277]. But the fitted observable in practice is
the strength DISTRIBUTION, not failure-location frequencies; fractographic failure-origin locations
are recorded (dental/ceramic fractography) yet rarely published as location-frequency datasets, and I
found no one fitting correlated weakest-link models to failure-location frequencies — a genuine gap,
but one without ready public data.
**(ii)** Data: sparse/proprietary; best bets are fiber-fragmentation tests and published fractography
tables.
**(iii)** Incumbent: two-parameter Weibull everywhere; deviations handled by mixtures or generalized
weakest-link forms (J Mater Sci 2017), not by dependence.
**(iv/v)** Our angle: Gaussian-copula weakest link — element strengths correlated through a
random-field factor (process batch, spatial low-rank modes); failure-location probabilities for all N
elements exactly, and the removal pass prices "drill out / reinforce this region" counterfactuals.
Demo would need a collaborator's fractography dataset, or synthetic validation against fiber-bundle
simulations (arXiv:cond-mat/0609650 review).
**(vi)** J. Mechanics and Physics of Solids, Int. J. Fracture, Phys Rev Applied.
**Note on nucleation:** droplet-microfluidic nucleation statistics currently SUPPORT independent
Poisson behavior (multiple crystals per drop remain Poisson; depletion zones decouple events) — no
documented demand for correlated first-nucleation models. Park here.

---

## Cross-cutting: the Thurstone/MNP bridge (decision science proper)

Not a science per se but the shared statistical substrate: multinomial probit choice probabilities
"depend on the number of choice alternatives, and therefore the computational cost increases
significantly as the number of choice alternatives increases" [snippet]; full-covariance MNP is "not
scalable to a large number of choice alternatives, though factor structures on the covariance matrix
can make the model scalable to large choice sets" [snippet, cf. arXiv:2007.13247, "Scalable Bayesian
estimation in the multinomial probit model"]. That literature reaches for GHK simulation even with
factor structure; our exact O(QNL) evaluation of ALL N probabilities plus the shared removal pass is a
direct drop-in and should be cited as methodological positioning in whichever science paper goes first.

---

## Sources (primary ones consulted)

- Heathcote & Matzke 2022, Curr Dir Psychol Sci — https://journals.sagepub.com/doi/full/10.1177/09637214221095852
- Leite & Ratcliff 2010, multiple-alternative decisions — https://pmc.ncbi.nlm.nih.gov/articles/PMC2805113/
- Brown & Heathcote 2008 LBA — https://www.ampl-psych.com/wp-content/uploads/2021/05/Brown-and-Heathcote-2008-The-simplest-complete-model-of-choice-response-tim.pdf
- Tillman, Van Zandt & Logan 2020 racing diffusion — https://link.springer.com/article/10.3758/s13423-020-01719-6
- Johnson & Kuhn 2013, Bayesian Thurstonian models in JAGS — https://link.springer.com/article/10.3758/s13428-012-0300-3
- Yeast whole-genome replication model 2025 — https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013066
- MCMs and replication timing — https://genome.cshlp.org/content/25/12/1886
- MATAC-seq origin efficiency — https://academic.oup.com/nar/advance-article/doi/10.1093/nar/gkad1022/7416811
- Kronos scRT — https://www.nature.com/articles/s41467-022-30043-x
- Initiation-factor levels and origin efficiency — https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1008430
- Rhind "f = m*a" replication-timing framework — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8872135/
- Gerrish-Lenski clonal interference — https://pmc.ncbi.nlm.nih.gov/articles/PMC1456385/
- Levy-style lineage tracking context: Good et al./LTEE — https://arxiv.org/pdf/1803.09995
- Germinal-center immunodominance model — https://arxiv.org/abs/2310.10966
- Cross-competition CD8 hierarchies — https://www.biorxiv.org/content/10.1101/2025.10.26.684631.full.pdf
- Alpha Factor extension, RESS 2026 — https://www.sciencedirect.com/science/article/abs/pii/S0951832025011640
- NUREG/CR-5485 CCF guidelines — https://nrcoe.inl.gov/publicdocs/CCF/NUREGCR-5485_Guidelines%20on%20Modeling%20Common-Cause%20Failures%20in%20PRA.pdf
- Weibull misfit evidence — https://www.researchgate.net/publication/245059292 ; https://arxiv.org/pdf/0901.3277
- Statistical models of fracture review — https://arxiv.org/pdf/cond-mat/0609650
- Scalable Bayesian MNP — https://arxiv.org/abs/2007.13247
- Droplet nucleation statistics — https://pmc.ncbi.nlm.nih.gov/articles/PMC2953805/
