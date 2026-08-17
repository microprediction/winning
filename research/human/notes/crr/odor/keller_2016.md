# Keller & Vosshall 2016 and Dravnieks 1985 — a 19-descriptor menu inside a 146-descriptor menu

## Citation
Keller A, Vosshall LB. "Olfactory perception of chemically diverse molecules." *BMC Neuroscience*
2016;17:55. Distributed as Pyrfume archive `keller_2016`.
Dravnieks A. *Atlas of Odor Character Profiles.* ASTM, 1985. Distributed as Pyrfume archive
`dravnieks_1985`.

## Domain and stimuli
Olfactory descriptor rating (not forced-choice identification). Keller & Vosshall: 55 subjects rating
960 stimuli (480 odorants at two concentrations) on **19 descriptors** plus intensity and pleasantness.
Dravnieks: expert panel applicability ratings on **146 descriptors**.

## Master response set and restricted response sets (nested, overlapping, or a relabelling)
**A genuine nesting — 19 descriptors inside 146 — over a shared stimulus set, but across studies and
30 years apart.**

65 odorants are common to both collections, and all 19 Keller descriptors have Dravnieks counterparts.
So there is a real T(19) ⊂ S(146) relation over 65 shared stimuli.

Why it cannot carry the argument:
1. **Different subjects, different eras** (1985 expert panel vs 2016 online/lab sample) — no
   within-subject comparison is possible.
2. **Dravnieks is aggregate-only in every public copy**: percent applicability and percent usage,
   panelist averages. Keller is fully raw. So it is aggregate-versus-raw, not like against like.
3. These are **rating/applicability tasks, not pick-one choices** — the response object is a vector of
   ratings, so a regularity test would need reformulating as "does a descriptor's applicability fall when
   135 more descriptors compete?"

## What numbers are printed or deposited (which tables/files, counts or proportions, per subject or pooled)
Keller: **fully raw and per-subject** — approximately 1.43 million rows (55 subjects × 960 stimuli × 19
descriptors). Dravnieks: **pooled only** (percent applicability / usage per odorant × descriptor).

## Access (a DIRECT url you fetched; open, paywalled, or Wayback-only)
https://github.com/pyrfume/pyrfume-data — **open**, public repository holding both archives
(`keller_2016/`, `dravnieks_1985/`). Keller & Vosshall 2016 is open access in *BMC Neuroscience*.
Note: the Pyrfume archive contents were identified from the repository listing; individual data files were
**not** downloaded and parsed in this session.

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)
**Usable now, but for a different question than ours.** Both are open and machine-readable. The
19-in-146 nesting is real but between-study and aggregate-on-one-side, so it cannot support a
within-subject menu-effect test. Keller's raw 19-descriptor matrix over 480 odorants is however an
excellent substrate if you ever want to *run* a nested-menu rating experiment: the master menu and
baseline shares already exist.

## What the authors concluded, quoted verbatim where possible
Keller & Vosshall's conclusion concerns the structure of olfactory perceptual space and the difficulty of
predicting percepts from molecular structure — individual variation is large and descriptor use is only
partly predictable from chemistry. Neither study addresses menu size; the nesting is an artifact of two
independent instrument designs.
