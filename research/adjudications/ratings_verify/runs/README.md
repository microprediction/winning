Committed verification reports, one JSON and one markdown per run,
named `<date>_<sha7|wheel>_<profile>`. Only runs that adjudicate
something are committed here (a baseline, the before/after pair around
a fix); regenerate any of them with

    python -m winning.ratings.verify --profile <profile> --workers 4 --out research/adjudications/ratings_verify/runs

at the commit named in the file's provenance block. The ledger is
`../ratings_verify.md`.
