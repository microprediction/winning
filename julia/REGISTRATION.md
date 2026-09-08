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
   version, [compat] on julia; a copy of the MIT LICENSE INSIDE each
   package subdirectory (AutoMerge does not look at the repo root);
   `MvNormalCDFFast`'s weakdep has an upper-boundable [compat] — add
   `MvNormalCDF = "0.2, 0.3"` style bounds when registering, the
   automerge bot requires compat entries for all deps.
4. Tags for subdirectory packages are named after the PACKAGE, not
   the path: `MultinomialProbit-v0.1.0`. Install TagBot
   (github.com/JuliaRegistries/TagBot), which gets the subdir naming
   right and only tags once a version is actually registered.

Order: MultinomialProbit and MvNormalCDFFast can register
independently; register `julia/winning` whenever its API settles
(blocks/tree/classic still on its roadmap).

## Updating a pending registration

Keep the version number the same and re-trigger on a newer commit:
the existing PR updates in place. Changing the version, name or repo
URL opens a NEW PR, and the old one then has to be closed by hand.
