# Telemetry / systems: the correlated latency race
(Three adjudication agents, 2026-09-03. Verdict: PURSUE narrow, at
the single-race level only; the whole-trace level is a structural
PARK. All package claims code-checked; external locators
agent-verified at the level noted.)

## The dividing line, checked
A latency MODEL already exists (measured per-backend histograms,
queueing or log-normal fits, plus a topology correlation story), and
operators need the distribution of its max/min -- both the tail
MAGNITUDE (p99/p999 of the fan-out) and the argmax/argmin IDENTITY
(which backend is the straggler; which hedge helps). Passes the
dividing line at the leaf-race level.

## CORRECTION (2026-09-03, Peter's pushback): first-failure is the
## stronger fit, and the paper already serves it
I dispatched three agents on LATENCY (the max race) and
under-weighted the MIN race. First-failure fixes every kill risk
that parked the other facets, and it is the engine's native
convention.

Peter's two points:
1. "First-failure races in systems." A system's time-to-failure is
   T_(1) = min_i T_i over components, and the CAUSE is the argmin --
   which drive, node, or resource goes first. This is competing
   risks, which is the paper's OWN native interpretation (main.tex
   line 137: G = prod S_k all-causes survival, f_i/S_i cause-specific
   hazard, p_i = int G (f_i/S_i) the classical cause-probability
   formula). The application the systems agents missed is the one the
   paper's math already is.
2. "Each caused by many possible paths." A component fails via many
   competing MODES (head crash OR bearing OR firmware OR ...), T_i =
   min over modes, and a SYSTEM fails via many cut sets (fault tree).
   "Many paths" is what makes n large and the correlation structured:
   paths share stressors (temperature, vibration, load, batch).

Why first-failure beats the latency facets:
- Native min-wins; no log-space transform, no max-of-sums recursion.
- The survival/competing-risks model already exists AND the paper
  claims it -- no modeling gap to argue.
- RARE events: you cannot estimate the tail of first-failure by
  counting, so the model is not fighting t-digest (the kill risk
  that parked facet 2 and bounded facet 1). This is the winning
  condition, not the losing one.
- Large n (Backblaze: 340k drives) is the engine's home scale.
- Correlation is MANUFACTURED and OBSERVABLE: same drive model/batch
  = same latent defect factor, and the model field is IN the data.
  Shared power/rack/vibration = block; datacenter = tree. The
  cleanest factor structure of any application considered.

## k-out-of-n and the count polynomial (the "many paths" machinery)
Erasure-coded storage loses data when the k-th correlated failure
occurs before repair: the system life is the k-th order statistic
T_(k), not just the min. P(T_(k) <= t) = P(N_t >= k) where N_t is
the failure count by t; conditional on the shared factors N_t is a
sum of independent Bernoullis (Poisson-binomial), and the
count-generating polynomial prod_i [S_i(t) + z F_i(t)] with
coefficient extraction is EXACTLY the shipped top-k machinery
(winning/factor/topk.py). So the engine already prices: time to
first failure (min), time to data loss (k-th), which component/mode
is the weakest link (argmin / criticality), and the removal/addition
counterfactual (retire the worst batch, add a parity drive -> the
new T_(k) distribution) -- all from one shared field, all native.

## Real benchmarks (verified/known 2026-09-03)
- Backblaze Drive Stats: PUBLIC, 2013 through Q1 2026, 340k+ live
  drives, 530M+ drive-days, daily CSVs with drive MODEL and
  manufacturer (the correlation factor, in the schema), failure
  flag, SMART covariates, date, serial. The gold-standard public
  first-failure dataset, with the factor variable built in.
- Computer Failure Data Repository (CFDR, USENIX): Los Alamos and
  other HPC node/component failure logs. [U -- verify current URL.]
- Schroeder & Gibson, "Disk failures in the real world" (FAST'07)
  and the DRAM-errors study (SIGMETRICS'09): the canonical analyses,
  and both document CORRELATED failures (batch, environment) that
  the independent-exponential MTBF model misses -- the reliability
  analogue of Slepian: independence mis-states the joint failure
  tail, and here the mis-statement is what the field already knows
  is wrong.

## Revised priority
First-failure / dependent k-out-of-n reliability is the FLAGSHIP
systems fit, above the latency facets: native convention, a model
that already exists and that the paper claims, rare-event regime
where measurement cannot compete, manufactured observable
correlation, home-scale n, and a gold-standard public benchmark
(Backblaze) with the factor variable in the data. Decisive
experiment: fit a factor survival model to Backblaze by
model/batch/vintage, predict the fleet's first-failure and k-th-
failure tails and the per-batch criticality, and beat the
independent-exponential (constant-hazard, uncorrelated) baseline the
field uses -- the same shape as the fan-out experiment, on data
where the correlation is labeled. This subsumes the earlier
weakest-link (Track C) and dependent-k-out-of-n research threads
into a systems venue with real data.

## The sharp boundary the three facets drew
- SINGLE fan-out / scatter-gather (one call group): total latency =
  max of n correlated backend latencies. A genuine race. STRONG fit.
- MIN of correlated replicas (hedging): adding a hedge = an addition
  counterfactual on the min. Exact correspondence.
- WHOLE TRACE (DAG of spans): latency = max-of-sums-of-maxes, a
  recursion of races and sums, NOT one order statistic;
  reconvergent fork-in is the SSTA rock, worse here. PARK.
The application is real but scoped to the leaf, not the trace.

## Facet 1 -- fan-out tail latency: PURSUE (narrow)
Dean & Barroso, "The Tail at Scale" (CACM 2013): their headline "63%
of requests over 1s" IS 1-(1-p)^n = 1-0.99^100 = 0.634 -- the
independence formula. They discuss shared-host/queue/GC correlation
only qualitatively, then pivot to mitigation. Downstream: EVT fits
marginal tails, queueing bounds one queue, t-digest/HDR MEASURE
realized percentiles, and where the correlated max is wanted it is
SIMULATED (Nguyen HotCloud'16). The correlated max distribution is
essentially never computed in closed form.
KEY FINDING, counterintuitive and useful: positive correlation
(shared host/rack) makes the max tail LIGHTER than 1-(1-p)^n --
stragglers cluster, fewer independent chances to be slow -- so the
ubiquitous independence formula OVER-provisions. A quantifiable
dividing line, not just "more accurate".
Covariance maps cleanly: shared host -> factor, rack/ToR -> block,
DC tree -> hierarchical, per-backend jitter -> diag(D).
Log-normal/Pareto service times (the load-bearing capability
question): no NAMED log-normal base ships, but the lattice
represents arbitrary smooth marginals and the race is invariant
under a common monotone transform, so log-normal is the Gaussian
case in LOG-SPACE (max of log-normals = exp(max of logs)) with
correlation specified there. Capability present as a transform.
Kill risk: t-digest wins for observed steady-state p99. The model
earns its place only at rare-tail SLOs (p999+, traces too sparse to
estimate), what-if capacity/fan-out/replication changes, and
hedging (argmax + counterfactual). If the customer only wants
realized p99 of the current topology, park it.

## Facet 2 -- trace critical-path criticality: PARK
Incumbents count deterministic critical paths across millions of
stored traces (Dapper 2010; Mystery Machine OSDI'14; CRISP ATC'22),
at Uber/Facebook scale. Structural killer: a trace is a max-of-sums
recursion, not one race, and reconvergent fork-join breaks the
factor grammar (the SSTA problem, worse). Abundant-data domain where
the incumbent is cheap counting -- the inverse of the engine's
winning conditions. Escape hatch (narrow): sparse-tail attribution
and the speed-up-service-X counterfactual, only once fork-join is
lattice-expressible.

## Facet 3 -- hedging / straggler decisions: PARK-leaning-probe
Every incumbent is a tuned threshold: Dean-Barroso hedge at the p95;
MapReduce backup tasks near completion; Spark speculation.quantile
0.75 x 1.5 median; LATE progress-rate; service-mesh retry budgets.
Only Mantri (OSDI'10) does cost-benefit (t_rem vs t_new), with NO
cross-replica correlation. Exact correspondences: how-many/which
hedges = the submodular group-selection optimizer
(research/selection); marginal hedge value = an addition
counterfactual on the min; when-to-speculate = the rollout-pruning
free boundary (research/design), generalizing Mantri's scalar to a
stopping boundary on a correlated race.
Two serious kill risks: (a) correlation must be estimable ONLINE in
the tail where data is sparsest -- estimate from topology, not
per-request; (b) THE LOAD-COUPLING GAP -- redundancy's real danger
is that extra requests raise utilization toward congestion collapse
(Vulimiri CoNEXT'13; Gardner SIGMETRICS'15), a queueing COST the
min-race does not model. The engine prices the hedge BENEFIT but not
the systemic cost, so alone it over-recommends hedges. A credible
tool must pair benefit (min shift) with cost (utilization) -- the
difference between a toy and a usable knob.

## The coherent project, if pursued
Facets 1 and 3 are one thread: the fan-out max model IS the input to
the hedge decision. Scope to a single call group where the structure
is a real race; model the max for provisioning/SLO and the
min-counterfactual for hedging; and PAIR the benefit with a
utilization cost so it does not over-recommend. That last point is
the systems-credibility bar the pure-race view misses.

## Decisive experiment (facet 1, the cleanest)
Alibaba cluster-trace-microservices-v2021 (github.com/alibaba/
clusterdata): pick fan-out call groups, fit VV'+D from co-located-
host structure with log-normal (log-space) margins, predict the
max-latency tail at p99/p999/p9999, and compare against BOTH the
empirical max tail AND 1-(1-p)^n. Win condition: the correlated max
tracks empirical within trace noise while independence
over-predicts by a demonstrable margin, and the argmax predicts the
realized straggler host. DeathStarBench as controlled cross-check.

## Priority
Genuinely novel (nobody computes the correlated fan-out max in
closed form) but NARROW and below exact_pom / eval-stats. Worth one
probe -- the Alibaba fan-out experiment -- which if it shows the
over-provisioning gap seeds a short note; the hedging decision layer
is the second step and needs the cost model to be credible.

## Real benchmark data (verified 2026-09-03)
Alibaba cluster-trace-microservices-v2021 carries EXACTLY the three
fields the fan-out experiment needs, confirmed from the schema:
- MS_CallGraph_Table (25 GB, 0.5%-sampled): traceID + rpcID give the
  fork-join call structure; rt is per-call latency in ms.
- nodeid gives HOST PLACEMENT -- the shared-host factor loading, the
  correlation the experiment turns on.
- MS_MCR_RT_Table: consumerRPC_RT / providerRPC_RT, the aggregate
  per-microservice response-time marginals.
- Scale: 10,000+ nodes, 20,000+ microservices, 20M+ call graphs, 12h.
Stream-and-reduce as with the pass@k rollouts (keep only the leaf
call groups' (rt, nodeid) tuples). DeathStarBench (SocialNetwork /
HotelReservation, Gan ASPLOS'19) is the controlled synthetic
cross-check where the topology and load are known exactly. Google
2019 Borg trace lacks span DAGs -- not usable for fan-out.

## Clean mathematical representation
Factor model in LOG-latency (log-normal service times become
Gaussian, correlation stated where it is natural):
    log L_i = mu_i + v_i' F + sqrt(d_i) eps_i,
    F ~ N(0, I_k),  eps_i ~ N(0,1),  Sigma = VV' + diag(D).
v_i is backend i's exposure to shared factors (a 1 in column h if it
sits on host h; rack and DC-tree columns for block/hierarchical).

Scatter-gather latency T = max_i L_i. The SLO-violation tail is the
complement of the max CDF, and conditioning on F makes it the
shared-field integral the engine computes in O(nLQ):
    P(T <= t) = E_F prod_i Phi((log t - mu_i - v_i'F)/sqrt(d_i)).
The straggler identity is the max-wins vector p_i = P(L_i = max_j
L_j), the cavity computation; the hedge benefit is the min
counterfactual P(min over a chosen set > t) = E_F prod (1 - Phi(.)),
with adding/removing a replica an addition/removal counterfactual.

The independence baseline everyone uses is the SAME integral with F
deleted (k=0): P(T > t) ~ 1 - prod_i (1 - p_i), i.e. 1-(1-p)^n when
homogeneous. The gap between the two is not empirical -- it is
SLEPIAN'S INEQUALITY (Slepian 1962): for jointly Gaussian log-
latencies, P(all log L_i <= log t) is monotone increasing in the
pairwise covariances. Hence positive correlation (shared host/rack,
v_i aligned) makes P(T <= t) LARGER and the max tail P(T > t)
SMALLER than independence predicts. "Independence over-provisions"
is therefore a theorem, not a measurement: the correlated max tail
is dominated by the independent one, and the engine computes the
exact value between the two bounds. That is the clean claim the
Alibaba experiment then confirms numerically.
