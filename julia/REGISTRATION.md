# Registering the Julia packages in General (Peter's checklist)

Three packages live as subdirectories of this repo: `julia/winning`,
`julia/MultinomialProbit`, `julia/MvNormalCDFFast`. The General
registry supports SUBDIRECTORY registration, so no repo split is
needed. Steps (owner-only, which is why this is a checklist and not
done):

1. Install the JuliaRegistrator GitHub app on microprediction/winning
   (https://github.com/JuliaRegistries/Registrator.jl — "install app").
2. On the commit to register, comment on GitHub:
       @JuliaRegistrator register subdir=julia/MultinomialProbit
   (one comment per package; `julia/winning` may draw a name-length
   nudge from the automerge bot — lowercase single-word names get
   flagged for manual review, not refusal; the R and PyPI precedent is
   the argument).
3. Requirements the packages already meet: Project.toml with uuid,
   version, [compat] on julia; OSI license in the repo (MIT);
   `MvNormalCDFFast`'s weakdep has an upper-boundable [compat] — add
   `MvNormalCDF = "0.2, 0.3"` style bounds when registering, the
   automerge bot requires compat entries for all deps.
4. Tag pattern after merge: TagBot handles it if installed; otherwise
   `git tag julia/MultinomialProbit-v0.1.0 && git push --tags`.

Order: MultinomialProbit and MvNormalCDFFast can register
independently; register `julia/winning` whenever its API settles
(blocks/tree/classic still on its roadmap).
