# Loomis 1990 — A model of character recognition and legibility

## Citation

Loomis, J. M. (1990). A model of character recognition and legibility. *Journal of
Experimental Psychology: Human Perception and Performance*, 16(1), 106–120.
doi:10.1037/0096-1523.16.1.106

(The DOI given in the brief, `10.1037/0096-1523.16.1.1066`, is a Crossref duplicate
carrying a mangled page range "1066-120"; both DOIs resolve to the same article. Verified
against the Crossref API, which returns both records.)

Full text obtained and read. This paper was not on the radar as a restricted-response-set
study and it turns out to contain one — **Experiment 3 is a clean nested 26 → 13 → 7
letter contraction** that the CRR literature appears never to have cited.

## Domain and stimuli

Character recognition, visual and tactile, in Loomis's programme comparing legibility
across the two senses. Four experiments plus reanalysis of Loomis (1981a, 1982).

Experiment 1: seven character sets (roman, braille, braille with small surround, katakana
and others; Sets 17–23 of the paper's Figure 1), presented tactually to the fingerpad
through a finger guide with 2-s contact, and visually through a glass diffuser acting as a
low-pass optical filter. Sets differ in character *type*, not as subsets of one another.

Experiment 2: uppercase roman letters on an Apple II / Video 100 monitor viewed through
the diffuser at four diffuser distances (60, 80, 106, 142 mm), giving four blur levels and
so four performance levels. Seven subjects, ten 2-hour sessions, each letter presented on
average four times per condition per session, feedback after each response.

**Experiment 3** — the relevant one. Same computer setup, same uppercase roman
characters, same viewing conditions and procedure as Experiment 2, with the *number of
available characters* as the manipulation.

Experiment 4: the seven sets of Experiment 1 again, direct foveal viewing at five viewing
distances instead of optical filtering.

## Master and restricted response sets

Experiment 3, p. 115, verbatim:

> "Five different sets of characters were used: the full set of 26, two subsets of 13
> (A through L and M through Z), and two subsets of 7 (A through G and M through S)."

So the master response set is the full 26-letter roman alphabet and there are four nested
restricted sets, two at size 13 and two at size 7, over the same stimuli, the same font,
the same display and the same blur levels. (The size-13 pair is printed as "A through L"
and "M through Z", which is 12 and 14 letters; Figure 4's legend labels them "A-L" and
"M-Z". Either the range or the count is a typo in the original. Anything downstream must
be based on the actual matrices, not on this sentence.)

Design detail that matters, p. 115:

> "The 20 paid subjects were divided evenly among the four subsets."

and

> "In the 7-character set conditions, subjects received eight presentations of each letter
> for each level of blur in each session. The number of presentations per stimulus was
> reduced to four for the 13-character set conditions and to two for the full set
> condition."

So each of the 20 subjects contributed both their assigned subset *and* the full 26-set —
master and restricted share subjects within a group — but **which subset a subject saw is
between-subjects**. Four sessions each, first session discarded as practice, data pooled
across subjects within condition. Crossed with four blur levels, so there are 4 blur
levels x 5 menu sizes = 20 condition matrices in principle.

Two design virtues worth recording. The restrictions are strictly nested, no relabelling.
And the two subsets at each size are *different* subsets of the same alphabet, which means
the same removal size can be tested twice with different survivors — a within-size
replication that no other study in this branch offers.

One design weakness: the 7-letter sets A–G and M–S are contiguous alphabet blocks, not
chosen for confusability, so whether a near-substitute of a survivor was removed is
accidental rather than controlled.

## What numbers are printed or deposited

**The Experiment 3 matrices were compiled but are not printed.** p. 115:

> "Confusion matrices were compiled for each of the conditions in the experiment; data
> from different subjects were pooled in the compilation, with the data from the first
> session excluded as practice."

What is printed for Experiment 3 is **Figure 4 only** — two panels, observed and predicted
percentage correct as a function of character set and stimulus bandwidth (cycles per
character height, at 0.37, 0.5, 0.67, 0.89), with the five sets as separate plotted series
labelled A-G, M-S, A-L, M-Z, A-Z. No table. No cell entries. Not even a table of the
percent-correct values behind the figure.

Elsewhere in the paper, also no matrices: Table 1 (p. 109) is percentage correct for the
seven sets x two modalities; Table 2 (p. 113) is product-moment correlations and rms
values for diagonal and off-diagonal cells of confusion matrices, i.e. goodness-of-fit
summaries; Table 3 (p. 115) is observed and predicted percentage correct for the four
conditions of Experiment 2 plus the same fit measures. The 26x26 matrices of Loomis (1982)
that feed Table 2 are in that earlier paper, not this one.

No deposit, no supplementary material, no data statement — as expected for 1990.

## Access with a fetched url

Fetched successfully:

    http://web.archive.org/web/20040507184332if_/http://www.psych.ucsb.edu:80/~loomis/loomis_90.pdf

HTTP 200, 1,264,099 bytes, 15 pages, scanned author reprint with a usable OCR text layer
(the published APA typesetting, pages 106–120). Located via the Wayback CDX index for
`www.psych.ucsb.edu/~loomis*`, which lists roughly forty Loomis reprints including
`loomis_81.pdf` and `loomis_82.pdf` — the two earlier papers whose matrices this one
reanalyses, and the more likely place to find printed 26x26 matrices.

Crossref metadata confirmed at `https://api.crossref.org/works?query.bibliographic=...`
(fetched). Semantic Scholar and OpenAlex both 404 on the correct DOI form.

## Usability verdict

**CRR-TEST-BUT-NUMBERS-NOT-PRINTED.**

The printed numbers do not suffice. Percent correct as a function of menu size cannot
score a full-menu-calibrated prediction against CRR, because CRR and any competing
contraction map are claims about the *distribution* of errors across survivors, and only
the diagonal is recoverable from Figure 4 — and even that only by reading points off a
figure. This is precisely the failure mode `../README.md` warns about: a study that varies
set size and publishes only percent correct.

Recorded anyway, because the file's job is to stop the next pass repeating the search.
Three things make it worth the entry:

1. **It is an unremarked 26 → 13 → 7 nested letter contraction with 20 subjects.** If a
   letter-identification data set with a large nested response-set restriction is ever
   wanted, this is the biggest one located in this branch. Loomis is emeritus at UC Santa
   Barbara; the matrices are 35-plus years old and almost certainly gone, but the design
   is on the record.

2. **Loomis's model is a set-size-dependent normalisation, and it reproduces the accuracy
   ordering.** His response-selection stage is Luce's unbiased choice model applied over
   whatever characters are in the present set, so shrinking the set mechanically raises
   predicted accuracy — the same arithmetic as CRR, run forward from template match
   strengths rather than from a master matrix. He reports it works on the diagonal (see
   quotes). That is a mild data point *for* renormalisation at the level of percent
   correct, and it is silent about the off-diagonal, which is where CRR is known to fail.
   Worth stating carefully if the project cites it: renormalisation getting the set-size
   effect on accuracy roughly right is compatible with it getting the distribution of
   errors wrong, and Townsend & Landon (1982) show exactly that combination.

3. **He is explicit that the model has no response-bias component**, so any residual is
   attributed to the stimulus-driven front end rather than to bias — which makes it a
   cleaner comparison than a biased-choice fit would be.

Category note: the paper never tests CRR and never mentions it in text (see below), so the
label is applied on the basis of the design plus the missing numbers, not on the basis of
an attempted test.

## Conclusion about CRR quoted verbatim

**The constant ratio rule is never discussed.** The only occurrence of the phrase in the
paper is in the reference list, in the title of Townsend & Landon (1982), which is cited.
Clarke (1957) is not cited. Searching the text for "constant-ratio" and "constant ratio"
returns that reference-list line and nothing else.

What the paper says about the normalisation rule that *is* CRR's decision content,
p. 112:

> "Given the various activations of each of the response alternatives, how does the
> subject go about selecting a response? A good descriptive model of response selection is
> the unbiased choice model of Luce (1963; Townsend, 1971): [Equation 6] The probability,
> P(i, j), of responding with label j given stimulus i is given by the fraction of total
> response activation accounted for by template j. The totality of probability values
> P(i, j) is referred to as a theoretical confusion matrix."

And immediately after, on the deliberate omission of bias, p. 112:

> "As was described earlier, the model is purely stimulus driven. If response bias were
> known to be a significant factor in the recognition process, then a model of response
> selection that correctly incorporates response bias would be desirable. Unfortunately,
> there is as yet little consensus on how it enters into response selection and on how to
> measure it (Appelman & Mayzner, 1981, 1982; Keren & Baggen, 1981; Loomis, 1982;
> Townsend, 1971; Townsend & Ashby, 1982; Townsend, Hu, & Kadlec, 1988). Accordingly, I
> have not attempted to include it in the model."

The motivation for Experiment 3, p. 114–115:

> "In all of the character recognition experiments reported so far, there have been 26
> characters in each set. Experiment 3 was conducted to determine whether the model,
> without modification, can account for the results obtained with character sets having
> fewer than 26 characters."

The result — the closest thing in the paper to a verdict on renormalisation across menu
sizes, p. 115:

> "Two other results are also important. First, the model accounts for the higher
> performance levels of the smaller character sets in relation to those of the full set of
> 26. Second, the model predicts rather well the ordering of performance between the two
> members of each pair of sets with the same number of characters and the same degree of
> blur; all 8 such order comparisons (four for the 7-character set and four for the
> 13-character set) were correctly predicted."

> "Experiment 3 shows that, without modification, the model accounts quite well for the
> results obtained with different character sets varying in number of characters, at least
> over the range studied. The model would probably fail in the case of a set of just two or
> three characters, for the subject would undoubtedly make more use of intensity variation
> in the stimulus that is assumed to be ignored in the model."

That last sentence is a hedge in the direction of the known CRR failure: Loomis expects
his normalisation to break down at very small menus, which is where Rouder (2004) and
Rouder (2001) find it breaking down.

On the model's overall limits with respect to off-diagonal structure, from the abstract,
p. 106:

> "Though purely stimulus driven, the model accounts quite well for differences in the
> legibility of character sets differing in character type, size of character, and number
> of characters within the set; it is somewhat less successful in accounting for the
> details of each confusion matrix."
