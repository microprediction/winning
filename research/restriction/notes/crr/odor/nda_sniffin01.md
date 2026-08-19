# NIMH Data Archive `sniffin01` — a schema that preserves the chosen descriptor

## Citation
National Institute of Mental Health Data Archive (NDA). Data structure `sniffin01`, "Sniffin' Sticks"
(status: Published, 98 elements). Also checked: data structure `bsit01` (Brief Smell Identification
Test). Not a publication — a data dictionary. Filed under the structure name for findability.

## Domain and stimuli
Olfaction. The NDA schema against which contributing studies submit Sniffin' Sticks data: 16-item
identification subtest plus 16 discrimination trials.

## Master response set and restricted response sets (nested, overlapping, or a relabelling)
**No restriction in the schema itself — fixed 4 alternatives per item.** Two features make it worth a
note:

1. `sniff16_1` … `sniff16_16` are coded **1::4 with the four descriptor words spelled out in the value
   labels** (e.g. `sniff16_16`: 1=Bread, 2=Fish, 3=Cheese, 4=Ham; `sniff16_2`: 1=Smoke, 2=Glue,
   3=Leather, 4=Grass). So any collection submitting against this structure yields per-subject chosen
   descriptors with names attached — no answer-key reconstruction needed, unlike `tolomeo_2026.md`.
2. A `sniff16_version` field coded **1 = Blue, 2 = Purple** exists. Since the blue and purple
   identification versions are two different 4-descriptor sets over the same odorants (see
   `delgado-losada_2021.md`), **any collection that administered both versions to the same subjects
   would be a same-odor / different-menu relabelling dataset with descriptor-level codes.** That is the
   most promising unexplored lead in this directory.

Discrimination trials `dis_1..dis_16` also record which pen was chosen — 3-alternative choices with a
known correct answer.

## What numbers are printed or deposited (which tables/files, counts or proportions, per subject or pooled)
Schema only — no data in the dictionary. Granularity is **per-subject, chosen descriptor**, by
construction.

**Critical contrast, and the structural reason a whole class of cohorts is hopeless:**
`https://nda.nih.gov/api/datadictionary/datastructure/bsit01` defines `bsit_1` … `bsit_12` coded
**0 = Incorrect / 1 = Correct only.** The standard B-SIT schema **discards the chosen descriptor**.
That single design decision is why ROS/MAP, Health ABC and every other B-SIT cohort in this directory
is a dead end regardless of access.

## Access (a DIRECT url you fetched; open, paywalled, or Wayback-only)
https://nda.nih.gov/api/datadictionary/datastructure/sniffin01 — fetched, **open**, returns the full
98-element definition with value labels, no login required.
https://nda.nih.gov/api/datadictionary/datastructure/bsit01 — fetched, **open**, confirms binary coding.
The **data** behind either structure requires an NDA account and a data-access request.

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)
**Schema usable now; data needs an NDA login and request. Worth roughly 20 minutes to scope.**
Nothing to digitize. The immediate task is to log in to NDA and query which collections have submitted
against `sniffin01`, then check whether any carries both `sniff16_version` values for the same subjects.
If one does, it is a ready-made relabelling dataset with named descriptors — better than anything else
found for that particular axiom test.

Cost is low, expected value uncertain (the number of contributing collections is unknown without a
login), so treat it as a cheap option rather than a plan.

## What the authors concluded, quoted verbatim where possible
Not applicable — a data dictionary carries no conclusions. The only editorial content worth recording is
the coding decision itself: `sniffin01` preserves which descriptor was chosen, `bsit01` does not.
