# Odor identification: response-set restriction survey

Search question: publicly downloadable odor-identification data where the **number of verbal response
alternatives** is manipulated, with the smaller set **nested** inside the larger — i.e. choice shares
over a descriptor set S and over a subset T of the same descriptors.

One file per experiment/source, each with the same seven headings. Negatives are included deliberately
so they are not re-searched.

## Verdict in one line
The nested manipulation exists in only three places — Negoias 2010 (olfaction, 3⊂6, no data),
SCENTinel (olfaction, 4→3 remaining, undeposited), and Jaeger 2015 (sensory CATA, 10–17⊂20–28,
undeposited). Everything that *is* downloadable uses a fixed four-alternative list.

## Usable now (open, fetched, per-subject chosen descriptor)
| File | What | n |
|---|---|---|
| `nhanes_2014.md` | NHANES CSX_H Pocket Smell Test, 8 items, codes 1–4 + published counts | ~3,520 × 8 |
| `jaen_2024.md` | NIH Toolbox via BHR, 9 items, descriptors as literal words | 1,163 × 9 |
| `tolomeo_2026.md` | Sniffin' Sticks SS-16, 4-way codes (descriptor names stripped) | 397 × 16 |
| `keller_2016.md` | 19-descriptor ratings over 480 odorants (different task) | 55 × 960 |
| `visalli_cata.md` | free-comment vs term-list, open CC BY (between-subjects caveat) | 60+436 |

## Needs an application (free, known quantity)
`mitchell_2025.md` (PPMI UPSIT, 40 items × chosen alternative, ~1 week DUA) ·
`kern_2014.md` (NSHAP, ICPSR registration) · `nda_sniffin01.md` (NDA login, ~20 min to scope)

## Needs digitizing from print
`singh_2025.md` (Table 3, complete 5-option shares, 16 odors — cheap and worth doing) ·
`hunter_2024.md` (fragmentary prose percentages)

## Needs library access before we can even judge
`raj_2023.md` (n=2,479, distractor choice is the paper's subject) · `jaeger_2015.md` (nested CATA) ·
`sulmont-rosse_2005.md` · `sorokowska_2015.md`

## Email targets, in priority order
1. Monell — Hunter / Reed / Dalton — SCENTinel first-and-second-choice records (`parma_2021.md`)
2. SNAC-K database committee — free-then-cued + distractor data (`horberg_2025.md`, `raj_2023.md`)
3. Hummel / Negoias — the 2010 crossover raw sheets (`negoias_2010.md`, `lotsch_2023.md`)
4. Ares / Castura — nested CATA arrays (`jaeger_2015.md`)
5. Parma / Tolomeo — the code→descriptor key (`tolomeo_2026.md`)

## Checked and dead — do not re-search
`ukbiobank.md` (no test exists) · `rosmap.md` · `aric.md` · `cardia.md` · `healthabc.md` ·
`tian_2023.md` · `schubert_2012.md` (20-option array, no access route) · `nakanishi_2024.md` ·
`brandao_2025.md` · `bastos_2015.md` · `ribeiro_2016.md` (deposit DOI is dead) · `eluecque_2015.md` ·
`capkan_2025.md` · `dalton_2013.md` · `lindroos_2022.md` (misleading data statement) ·
`italian_sset_osf.md` · `hummel_norms.md` · `sorokowska_2014.md` · `liu_2020.md` (11-option list but
not nested, mixtures, restricted)

## Two structural facts worth remembering
- **The B-SIT data model discards the chosen descriptor** (0/1 only), which is why every B-SIT cohort
  is hopeless regardless of access. The Sniffin' Sticks NDA schema keeps it. See `nda_sniffin01.md`.
- **NHANES and UPSIT menus differ by exactly one substituted distractor on three odors** — a
  single-element swap, arguably a sharper IIA probe than nesting. See `mitchell_2025.md`.

## Caveats carried forward
- `negoias_2010.md`: the 22-of-32 nesting list comes from HTML extraction that proved unreliable.
  Certify against the print PDF before use.
- `tolomeo_2026.md`: the only copy of that data hangs on a revocable anonymized OSF token. **Archive it.**
