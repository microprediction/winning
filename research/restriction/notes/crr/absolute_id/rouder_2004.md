## Citation

Rouder, J. N., Morey, R. D., Cowan, N., & Pfaltz, M. (2004). Learning in a
unidimensional absolute identification task. *Psychonomic Bulletin & Review*, 11(5),
938-944. DOI 10.3758/BF03196725.

## Domain and stimuli

Absolute identification of line lengths, with extended practice. Set sizes of 13, 20
and 30 stimuli.

## Master response set and restricted response sets (nested, overlapping, or a relabelling)

**A respacing, not a nesting.** Verified from the experiment source in the data
repository. The three set sizes use different physical length arrays:

    LT2.C   int length[13]={15,23,32,43,57,73,93,116,143,174,210,251,298};
    LT3.C   int length[20]={9,12,16,20,26,33,41,50,60,72,85,100,117,136,157,180,206,234,264,298};
    LT4.C   int length[30]={9,12,14,17,20,23,27,31,36,41,47,53,60,67,76,84,94,...};

Each set is independently power-spaced across roughly the same total range, so the 13-set and
20-set share only the endpoint 298. There is no common alternative set to restrict, and
responses are within-set ranks. Neither nested nor overlapping in any usable sense.

## What numbers are printed or deposited (which tables/files, counts or proportions, per subject or pooled)

Raw trial-level data **is** deposited, which is why this is worth recording rather than
dismissing. Path `1dMemory/LONGTERM`, one file per subject, 7 columns:
`sub blk trl stim resp second_resp RT_ms`, where `second_resp` is -1 when no second response
was given. Verified `LT3M00` has 720 trials. The length arrays above match the paper's
stimulus table.

## Access (a DIRECT url you fetched; open, paywalled, or Wayback-only)

**Open**, no login. Fetched, HTTP 200:

    https://raw.githubusercontent.com/PerceptionCognitionLab/data0/master/1dMemory/LONGTERM/LT3M00

Directory: `https://github.com/PerceptionCognitionLab/data0/tree/master/1dMemory/LONGTERM`

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**Unusable for this project** — excellent open raw data, wrong manipulation. Set size
varies but the alternatives are not shared across conditions, so there is no restricted
response set over a common master. Recorded so nobody re-opens it hoping otherwise.

Same repository, different subdirectory, does have what we need: see
`rouder_unpublished_chunk.md`.

## What the authors concluded, quoted verbatim where possible

Not read in full; the design was established from the experiment source rather than the
paper. Nothing quoted.
