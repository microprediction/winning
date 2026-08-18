# Jones, Wills & McLaren (1998)

## Citation

Jones, F.W., Wills, A.J., & McLaren, I.P.L. (1998). Perceptual categorization:
Connectionist modelling and decision rules. *The Quarterly Journal of Experimental
Psychology B: Comparative and Physiological Psychology*, 51B(1), 33–58. University of
Cambridge. (The Wills et al. 2000 reference list gives "51B(3), 33-58"; the article's own
running head gives 51B(1).)

The immediate predecessor to Wills et al. (2000), and cited there as the earlier
demonstration that "the ratio rule can be rejected" in categorization.

## Domain and stimuli

Human perceptual categorization of prototype-structured artificial visual stimuli — the
same stimulus family later used in Wills et al. (2000): 12 different symbols randomly
arranged on an invisible 4x3 grid, drawn from a pool of 36 symbols. Non-speech, non-letter.

Observational training (subjects watch labelled exemplars rather than getting feedback),
then a categorization test over a graded series of transfer stimuli spanning the two
prototypes (proportion of B symbols from 0/12 to 12/12).

Experiment 1: 32 subjects. Experiment 2: two further groups differing in amount of
training (30 vs 10 exemplars per category).

## Master response set and restricted response sets (nested, overlapping, or a relabelling)

Yes, but the manipulation is of the **trained** label set versus the **test-available**
label set, not two nested test sets over the same trained categories. Group notation is
Training/Test:

- **A30 B30 / AB** — two categories trained (30 exemplars each), test response set {A, B}.
- **A30 B30 / ABX** — identical training; test stimuli have half their elements replaced by
  never-seen elements (a generalization-decrement, i.e. evidence-suppression, manipulation
  analogous to Wills et al.'s novel-elements condition). Compared *within subject* against
  A30 B30 / AB because training was identical.
- **A15 B15 C30 / AB** — three labels used in training (half of each category's exemplars
  mislabelled as a third category C), but the **test response set is only {A, B}**: C is
  trained and then removed from the response set. Different subjects.
- Experiment 2: A30 B30 / AB* and A10 B10 / AB* (denser 13-point test series).

So the C category in A15 B15 C30 / AB is a trained alternative that is **excluded at test**
— exactly the configuration on which the ratio rule's renormalization has to be right.
The paper's own conclusion on this point is a statement that the decision rule must
renormalize over *available* alternatives only.

Decision rules compared, all fitted to the same data: simple ratio rule
P(a) = A_a / sum A_j; exponential ratio rule P(a) = e^{kA_a} / sum e^{kA_j} (i.e. Luce
softmax); simple difference rule; exponential difference rule; and a noisy winner-take-all
network (Thurstonian competitive race).

## What numbers are printed (which tables, counts or proportions, per subject or pooled)

No confusion matrices; no data tables. Everything is in figures:

- Figure 1: stimulus symbol pool and an example stimulus.
- Figure 2: mean response probabilities (P("A")) and mean latencies for groups
  A30 B30 / AB, A30 B30 / ABX and A15 B15 C30 / AB, as a function of proportion of B symbols.
- Figures 3, 4, 6, 7: predicted P("A") curves for the simple ratio rule, exponential ratio
  rule (k = 4.4), simple difference rule (k = 1.4) and exponential difference rule
  (k = 0.8), overlaid against the data with error bars at the 5% significance level
  (Bonferroni-corrected for the number of comparisons).
- Figure 11: mean response data for A30 B30 / AB* against winner-take-all predictions.
- Figure 12: mean number of WTA cycles to decision, averaged over 1,000 simulations.
- Inferential statistics are reported inline (e.g. F(1,15) = 17.40 for the ABX effect).

Pooled means across subjects, with per-subject fitting used to obtain the k values.
Raw data do **not** appear in the willslab data archive (only cam1 = Wills et al. 2000
Experiment 2 is deposited).

## Access (a DIRECT url you have fetched; open, paywalled or Wayback-only)

Open, author-hosted full text (fetched, 411 KB, 27 pages, complete article):
https://www.andywills.info/assets/pdf/1998jones.pdf

Publication list confirming the file path (fetched):
https://www.andywills.info/publications

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**Needs digitizing, and only partially useful.** The full text is free, but every data
point lives in a figure — there are no printed proportions and no deposited raw data. To
use it you would have to digitize Figure 2 (three curves x 7 stimulus levels) and Figures
7/11 (13 levels). Even then it is not a master/subset confusion-matrix study: the response
set is binary {A, B} in every test condition, so you cannot form an odds ratio between two
survivors under two different menu sizes. Its value is as **corroborating precedent**: it
shows within the same paradigm that (a) the simple ratio rule is quantitatively rejected,
(b) an exponential (softmax) ratio rule survives only if it renormalizes over the
test-available alternatives, and (c) a noisy winner-take-all/Thurstonian race does equally
well and additionally predicts latencies. Wills et al. (2000) is the paper to mine for data.

## What the authors concluded, quoted verbatim where possible

From the abstract (ligature-restored):

> "Although it is currently popular to model human associative learning using connectionist
> networks, the mechanism by which their output activations are converted to probabilities
> of response has received relatively little attention. Several possible models of this
> decision process are considered here, including a simple ratio rule, a simple difference
> rule, their exponential versions, and a winner-take-all network. ... Only the exponential
> ratio rule and the winner-take-all architecture, acting on the networks' output
> activations that corresponded to responses available on test, were capable of fully
> predicting the mean response results. In addition, unlike the exponential ratio rule, the
> winner-take-all model has the potential to predict latencies."

Experiment 1 discussion — the response-set point:

> "The fact that the mean response results for groups A30 B30 / ABX and A15 B15 C30 / AB were
> very similar suggests that, at least in this case, decision rules should only include the
> activations of output units that correspond to responses available on test."

On the simple ratio rule:

> "From Figures 3 and 4 it is clear that the simple ratio rule is the only function
> inconsistent [with the data]" ... "In summary, the evidence converges on the conclusion
> that the simple ratio rule does [not hold]."

Conclusions section, on preferring the Thurstonian race:

> "Only the exponential ratio rule, operating on the activations of the name units that
> corresponded to responses available on test, was able to approximate to an adequate
> account of all the mean response data presented here. A noisy winner-take-all
> architecture was also found to be compatible with the mean response results. This later
> approach has the advantage of providing a mechanism whereby a response is made; there is
> something unsatisfactory about using a rule that gives a probability of response, then
> throwing a dice to decide the outcome. Furthermore, in contrast to many theories of
> categorization, this mechanism has the potential to generate latencies as well."
