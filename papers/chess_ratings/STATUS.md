# Status: under correction (2026-09-13)

The numbers in `paper.tex` are superseded. Do not quote them.

The comparison tuned Glicko-2's `tau` on a validation split and cited
that as evidence that neither side carried a free parameter the other
lacked. In that implementation `tau` enters only the volatility
iteration, volatility barely moves over one month, and a sweep from
0.02 to 1.0 returns bit-identical held-out loss. The selection was a
tie-break on the first grid entry, and `period` and `initial_rd`, which
do move the loss, were never swept. Our own arms were tuned
asymmetrically too, with the level ridge pinned at 1.0.

Retuning both sides symmetrically (bandits exp41,
`research/chess/results/exp41_output.txt`):

| row | published | corrected | change |
|---|---|---|---|
| headline | −0.0160 | −0.0103 | 36% smaller |
| estimator | −0.0199 | −0.0080 | 60% smaller |
| structure | −0.0035 | −0.0033 | unchanged |
| stratification's value to Glicko-2 | +0.0074 | −0.0011 | sign flip |

The result survives: the factor arm still beats Glicko-2 per time
control on all three splits with intervals excluding zero. Three claims
do not. The estimator column and the four-fifths split are wrong; the
section arguing that stratification helps a weak estimator rests on a
number that has changed sign, and with both sides tuned stratification
helps neither; and the fairness sentence in the protocol section is
false as written.

Glicko-2's `period` selects at the top of its grid on every split in
the corrected run, so the corrected headline is itself an upper bound
on the margin.

A replacement draft is being prepared against the corrected runs, along
with exp40, which compares online against online and speaks to how much
of the estimator column is batch privilege. The guard that catches this
class of defect now ships as `winning.ratings.tuning`.
