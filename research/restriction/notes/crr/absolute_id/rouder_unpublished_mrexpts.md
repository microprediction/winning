## Citation

Unpublished lab data, Perception & Cognition Lab (Jeff Rouder), University of Missouri.
Repository `PerceptionCognitionLab/data0`, path `1dMemory/mrExpts`, experiments MR19-MR23
and mr0-mr17. No publication located. Same provenance caveat as
`rouder_unpublished_chunk.md`.

## Domain and stimuli

Absolute identification of visual line lengths. 12 stimuli per block drawn from a
16-length master. From `MR20.C`:

    int length[16]={9,12,16,20,26,33,41,50,60,72,85,100,117,136,157,180};

MR22 and MR23 use a shifted master, `{9,12,26,33,41,50,60,72,85,100,117,136,157,180,206,234}`.
Six 1-hour days per subject in MR20 (`day0`..), 540 trials in the file I checked.

## Master response set and restricted response sets (nested, overlapping, or a relabelling)

**Overlapping menus over a common universe, expressed as a relabelling — and recoverable.**
All three experiments define three 12-element subsets of the same 16-length master:

    int lengthsa[12]={2,3,4,5,6,7,8,9,10,11,12,13};
    int lengthsb[12]={0,1,4,5,6,7,8,9,10,11,14,15};
    int lengthsc[12]={0,1,2,3,6,7,8,9,12,13,14,15};

Only `a` and `b` are actually used — `stimset` is computed modulo 2 — and they alternate
*within* subject (by day in MR20, by half-session and subject number in MR22/23).

The logged `stim` and `rsp` fields are local indices 0-11 into `mylengths`, not global
indices, so on its face this is a relabelling. But because both menus index the same fixed
`length[16]` array and the mapping is order-preserving and known from the source, it inverts
exactly: local rank r in set X denotes global length `lengthsX[r]`. So P(choose alternative |
menu) is fully recoverable.

Neither menu contains the other, so this is **overlapping, not nested**. The two menus share
8 alternatives, global indices {4,5,6,7,8,9,10,11}; `a` adds {2,3,12,13} and `b` adds
{0,1,14,15}. That is an add/remove-alternatives comparison rather than a pure restriction.

## What numbers are printed or deposited (which tables/files, counts or proportions, per subject or pooled)

Raw trial-level data, one file per subject per experiment. 9 fields, self-labelling:

    sub001 day0 blk0 trl000 set0 stim9 rsp7 0 03973

from `sprintf(outline, "sub%03d day%i blk%i trl%03d set%i stim%i rsp%i %i %05d\n", ...)` —
so subject, day, block, trial, which stimulus set, stimulus (local), response (local),
correct flag, RT in ms. MR20 has 5 subjects with 1-5 files each (`DC`, `JZ`, `AW` initials).

## Access (a DIRECT url you fetched; open, paywalled, or Wayback-only)

**Open**, no login. Fetched, HTTP 200:

    https://raw.githubusercontent.com/PerceptionCognitionLab/data0/master/1dMemory/mrExpts/MR20/MR20DC01
    https://raw.githubusercontent.com/PerceptionCognitionLab/data0/master/1dMemory/mrExpts/MR20/MR20.C

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**Usable now, secondary priority.** Genuinely usable once the local-to-global remap is
applied, and it is within-subject, which is valuable. Two reasons it ranks below the `chunk`
data: there is no full 16-alternative master condition, so the calibration input has to be
one 12-menu predicting the other rather than a master predicting restrictions; and the menus
differ at both ends simultaneously, so a failure is harder to localise. Best use is as a
held-out overlapping-menu check on a map calibrated elsewhere.

Note MR22/MR23 use a different master array than MR20, so do not pool across experiments
without rechecking `length[]`.

## What the authors concluded, quoted verbatim where possible

No publication, no README. Nothing to quote.
