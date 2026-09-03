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
