# Fresh read of the manuscript, 19 August 2026

Read `papers/thurstone_humans/paper.tex` end to end against the notes. Two substantive
issues, one omitted literature, and two count inconsistencies. Ordered by how much a referee
would make of them.

## 1. The mechanism section contradicts the paper's cleanest dataset

Section `sec:mixture` derives Case V as the high-heterogeneity limit of a population of Luce
choosers whose tastes differ, and then draws the consequence: "That mechanism also says where
the fit should be worst: in populations whose members are nearly alike."

Section `sec:townsend` reports the Townsend and Landon result and says, correctly, that
calibration row and target row come from the **same subject**, so "the aggregation argument of
Section~\ref{sec:mixture} is not available as an explanation of any gain found here."

Both statements stand, and together they say the mechanism predicts the race should do worst
exactly where the paper's most controlled result has it winning at p = 0.005. A referee will
put those two sentences side by side.

**The fix is a rewording, not a retraction, and it strengthens the section.** The mixture in
Proposition~\ref{prop:mixture} is indexed by $\theta$, and nothing in the proof requires
$\theta$ to be a *person*. It requires a per-alternative Gaussian component that varies across
the occasions being aggregated. Trial-to-trial fluctuation in the perceptual representation of
each alternative supplies exactly that within one observer: a letter's evidence varies from
flash to flash. So the mixture index is the occasion, of which "different people" is one case
and "the same person on different trials" is another.

Read that way the mechanism covers the perceptual datasets at all, which on the current wording
it does not, since there the population is one observer's trials. It also keeps the prediction
that has empirical content: the fit should be worst where the aggregated units are nearly
alike *in the relevant sense*, meaning little occasion-to-occasion variation in the
per-alternative evidence, not few people. Concretely, "between individuals" and "populations
whose members are nearly alike" should become occasion language.

## 2. The operant literature is absent, and it contains a fifth boundary condition

The paper states four boundary conditions. The operant matching literature supplies a fifth,
orthogonal to all of them, from a matched pair of experiments in the same laboratory:

- **Elliffe & Davison (2010)**, *Behavioural Processes* 84:381-389, "Four-alternative choice
  violates the constant-ratio rule". Six pigeons, four-key concurrent VI, reinforcer
  distribution 27:9:3:1, assignment changing **every 10 reinforcers**. Pairwise preference
  depended "not only on the relative reinforcer rates on those keys, but also on the absolute
  levels of those rates."
- **Bensemann, Lobb, Podlesnik & Elliffe (2015)**, *JEAB* 104:7-19. The same six-pigeon,
  four-alternative, 27:9:3:1 design with each assignment held for **50 sessions**, and the CRR
  is **satisfied**.

The violation localises to the regime where the subject is still inferring the structure. That
is a fifth condition and it cuts both ways: it is a caveat on when the axiom fails and equally
a statement of when renormalization is fine. It also lands squarely on the paper's own machine
learning framing, where a masked softmax in a model still fitting is not the same object as one
at convergence.

Two honesty requirements come with it. Neither experiment restricts a menu — all four keys stay
available throughout, and the test is whether the odds between two survivors depend on the rest
of the set. That is evidence about the axiom, not about a transport map, and it should be
labelled as such rather than folded into the scoreboard. And these are pigeons.

**A fairness correction follows.** Section `sec:prior` asserts that "every claim of conformity
that was later retested with a test that had power was overturned." That is true of the
perceptual identification literature, which is what the sentence cites. It is not true of
operant choice: Davison and Hunter (1976) reported conformity, and Bensemann et al. (2015)
retested the same design at steady state and confirmed it. The sentence should be scoped to the
literature it describes.

## 3. Davison and Hunter (1976) is not scoreable, and the reason is worth one line

Table 1 prints responses, times and obtained reinforcers on all three keys for 27 conditions,
giving roughly nine matched three-key to two-key pairs. It is nonetheless not a scoreboard
candidate, for reasons that are all visible in advance:

- aggregate over six birds, and the average silently mixes Bird 143 with its replacement 143b;
- removal is by **extinction**, so the dead key stays lit and takes 0 to 101 pecks per
  condition, which forces an explicit and arbitrary coding decision;
- **obtained** reinforcer ratios shift across the matched conditions, so the worths change with
  the removal. That is the paper's own quality-changing-removal boundary condition, and it
  disqualifies the design rather than producing an interesting loss.

One anomaly is worth recording since the authors do not mention it: in the matched family that
deletes Key 3, the Key1:Key2 response ratio goes 1.12 to 0.87 at X = 120, a log-ratio sign
reversal, inside a paper concluding "strong support to Luce's principle." A sign reversal is the
Scottish verdict failure mode, and no contraction map handles it either.

## 4. Two count inconsistencies

- The abstract says "thirty-four population comparisons"; the scoreboard section says
  "Thirty-nine comparisons" and "thirty of the thirty-nine rows". Both may be defensible if 34
  counts distinct populations and 39 counts table rows, but as written they read as an error.
- The census appendix says "None of the eleven entries is a preference task" above a table with
  nine rows.

A verification pass is running on both, along with the rest of the arithmetic.
