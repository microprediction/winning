# Rouder 2004 — Effects of choice-set size on the processing of letters and words

## Citation

Rouder, J. N. (2004). Modeling the effects of choice-set size on the processing of
letters and words. *Psychological Review*, 111(1), 80–93.
doi:10.1037/0033-295X.111.1.80

Full text obtained and read. This is the most direct modern test of the constant ratio
rule in the visual letter literature, and it is a **prior-art hit**: the rejection of
proportional renormalisation on restricted menus, with a signed residual, is already in
print in *Psychological Review*.

## Domain and stimuli

Three data sets, all letter or word identification of briefly presented, masked stimuli:

1. **Rouder (2001) Experiment 2**, reanalysed. Letters Q, W, E, R, T, Y in a six-choice
   condition; letters W and E in a two-choice condition. Forward and backward symbol
   masks. 15 participants (14 usable for the ratio test — one had no errors in the
   six-choice condition, making the ratio undefined). Choice set manipulated across
   blocks of 50 trials, 16 blocks per participant in a single session.
2. **Townsend & Landon (1982)**, reanalysed. Letters A, E, F, H, X. Rouder uses the
   five-choice master and the two three-choice subsets (A,E,X and F,H,X); he does not use
   the four-choice A,E,F,H subset. 4 participants, 16 sessions.
3. **A new experiment of his own**, word stimuli. Four-letter words, frequency-matched
   pairs, briefly presented and masked, 12 participants. Conditions: a **naming**
   condition (open verbal response, treated as "a choice from about 1,200 alternatives"),
   a prestimulus two-choice cue condition, and a poststimulus two-choice cue condition.

## Master and restricted response sets

For the two reanalysed data sets, yes and nested — six-choice → two-choice
(Q,W,E,R,T,Y → W,E) and five-choice → three-choice (A,E,F,H,X → A,E,X and → F,H,X),
within subject in both cases.

For Rouder's own word experiment, **no usable master matrix exists**. The wide condition
is open naming over roughly 1,200 four-letter words, so there is no full-menu response
distribution to calibrate on — only accuracy. The restricted condition is a two-choice
menu of one target and one frequency-matched foil. So his own experiment is a
percent-correct comparison across menu sizes, not a matrix-to-matrix contraction test.

The load-bearing point for this project: **the CRR test in this paper is performed only
on letter-pair odds ratios, not on whole matrices.** For every letter pair {i,j} that
survives into a restricted menu, Rouder forms the odds P(i|stimulus)/P(j|stimulus) in the
wide condition and in the narrow condition, takes logs, and plots narrow against wide.
CRR predicts the points lie on the identity line. That is precisely the odds-invariance
content of CRR, tested pair by pair. Rouder (2001) data give 28 such ratios; Townsend &
Landon give 6 letter pairs x 2 ratios per pair = 12 ratios per participant, 48 over the
4 participants.

## What numbers are printed or deposited

**The paper contains no tables at all.** Every result is a figure: Figure 2 (log odds
ratios, the CRR test), Figure 3 (SCM distance estimates), Figures 4–7 and 9–11 (detection
estimates, all-or-none and IAM accuracy predictions, model fits). Only the counts of
points above the diagonal are given in text — 21 of 28 for Rouder (2001), 39 of 48 for
Townsend & Landon (1982).

The matrices are held back, with an offer:

> "The frequency matrices from the analyses may be obtained from the author."
> (author note, p. 80)

Written in 2003, so this is a 20-plus-year-old offer to a since-relocated author; treat
as not deposited. There is no supplementary file, no OSF or website deposit. Semantic
Scholar reports the paper CLOSED with an empty openAccessPdf url; the copy read here is
an author reprint recovered from the Wayback Machine.

For one of the two reanalysed data sets this does not matter at all: the Townsend &
Landon (1982) matrices are printed in full in the original paper (see
`townsend1982.md`), so that half of Rouder's analysis is fully reproducible from primary
print without contacting anyone. The Rouder (2001) 6x6 and 2x2 matrices are the ones that
exist only as a request.

## Access with a fetched url

Fetched successfully:

    http://web.archive.org/web/20151003091509if_/http://pcl.missouri.edu/sites/default/files/rouder.psyrev.2004.pdf

HTTP 200, 240,312 bytes, 14 pages, born-digital PDF with a clean text layer — the
published APA typesetting, pages 80–93. The live `pcl.missouri.edu` host no longer
resolves (curl returns exit 0 bytes / HTTP 000).

Metadata cross-checked at
`https://api.semanticscholar.org/graph/v1/paper/DOI:10.1037/0033-295x.111.1.80?fields=title,abstract,year,authors,openAccessPdf,externalIds,venue`
(fetched; abstract elided by publisher, status CLOSED, PubMed 14756587).

## Usability verdict

**CRR-TEST-BUT-NUMBERS-NOT-PRINTED.**

This paper cannot itself supply data for scoring a full-menu-calibrated Gaussian
prediction against CRR, because it prints no cell-level numbers for any of its three data
sets. Its value to the project is different and considerable:

1. **It is prior art on the negative result.** CRR is tested, on human restricted menus,
   in *Psychological Review*, and rejected — twice, on two independent letter data sets.
   Any claim that renormalisation fails on real restricted response sets has to cite
   this. Note the framing difference: Rouder reads the failure as a claim about people
   ("somewhat-efficient conditioning" on the available choices), not as a claim about the
   contraction map. The project's framing — that the residual is what a parameter-free
   Gaussian race predicts — is not tried here, and Rouder's own list of candidate
   explanations (Section "Decision models", general-recognition-theory variants) is where
   a Thurstonian answer would sit.

2. **The residual has a stated sign, and it is consistent across data sets.** Observed
   performance on the restricted menu is *worse* than CRR predicts; equivalently the
   log-odds points lie above the identity line, meaning the odds separating a survivor
   pair shrink when the menu shrinks. This is the same direction Rouder (2001) reports as
   "psychological distance between letters increased with an increased number of
   to-be-identified stimuli". Any Gaussian-race scoring on Townsend & Landon's printed
   matrices should be checked against this sign before anything else — if the Gaussian
   map does not move the prediction in this direction it is not explaining the known
   residual.

3. **It rules out several rival explanations in advance.** Because CRR is the decision
   rule inside FLMP, Keren & Baggen's recognition model, and SCM under invariant
   parameters, the same rejection lands on all of them (pp. 82–83); and the opposite
   failure is documented for models that use choice-set restrictions only when guessing
   (McClelland & Rumelhart's IAM, and the all-or-none model), which *underestimate* the
   use of restrictions. So the true contraction sits strictly between renormalisation and
   guess-only, which is a useful bracket to state.

4. **It supplies two leads.** Takane & Shibayama (1992), "Structures in stimulus
   identification data", in F. G. Ashby (Ed.), *Multidimensional models of perception and
   cognition*, pp. 335–362 (Erlbaum), is cited as having "provided more stringent
   statistical tests of the constant ratio rule" and rejected it — a book chapter, not
   checked in this sweep, worth a look for reanalysed matrices. Smith, J. E. K. (1992),
   "Alternative biased choice models", *Mathematical Social Sciences* 23, 199–219, is
   cited as the competitor model in the same territory.

If the Rouder (2001) frequency matrices could still be obtained they would add a 6→2
nested contraction with 15 subjects, which is a much larger removal than anything in
Townsend & Landon. Worth one email; not worth blocking on.

## Conclusion about CRR quoted verbatim

On Rouder's (2001) letter data, p. 82:

> "If the data obey the constant ratio rule, then the logarithms of the ratios should not
> vary with the number of choices, and the plotted points should cluster around the
> diagonal. However, most of these points (21 out of 28) are above the diagonal.
> Therefore, the constant ratio rule does not hold. Performance was worse in the
> two-choice condition than predicted by the constant ratio rule. The interpretation is
> that participants did not fully condition their identification on the reduced number of
> choices."

On Townsend & Landon's (1982) data, p. 82:

> "As can be seen, most of the points (39 out of 48) lie above the diagonal. This result
> indicates that participants did worse with fewer alternatives than would be expected
> from the constant ratio rule. The same conclusion is reached as with Rouder's (2001)
> data — participants were not fully efficient in conditioning their identification on
> the choice-set restriction."

On the reach of the rejection, p. 82 (of FLMP) and p. 83 (of Keren & Baggen):

> "Hence, the model is formally equivalent to the constant ratio rule and is challenged by
> the present analysis of Rouder's (2001) and Townsend and Landon's (1982) data."

> "Because similarity is a perceptual parameter that should be invariant to the
> choice-set-size manipulations, Keren and Baggen's model is challenged by the present
> analysis."

Overall conclusion, p. 92:

> "The main goals of this study were to assess the degree to which participants use
> choice-set restrictions in letter identification and to use this assessment to test
> decision mechanisms in identification models. The result is that participants are
> somewhat efficient in their conditioning on choice-set restrictions — they are neither
> as efficient as they would be by using ideal conditioning nor as inefficient as they
> would be by simply using choice-set restrictions when guessing. This intermediate
> result held for both letter and word identification."

And the abstract, p. 80:

> "Letters and words are better identified when there are fewer available choices. How do
> readers use choice-set restrictions? By analyzing new experimental data and previously
> reported data, the author shows that Bayes theorem-based models overestimate readers'
> use of choice-set restrictions. This result is discordant with choice-similarity models
> such as R. D. Luce's (1963a) similarity choice model, G. Keren and S. Baggen's (1981)
> letter recognition model, and D. W. Massaro and G. C. Oden's (1979) fuzzy logical model
> of perception. Other models posit that choice restrictions affect accuracy only by
> improving guessing (e.g., J. L. McClelland & D. E. Rumelhart's, 1981, interactive
> activation model). It is shown that these models underestimate readers' use of
> choice-set restrictions. Restriction of choice set does improve perception of letters
> and words, but not optimally."
