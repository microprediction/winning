# Performance densities from cumulative scoring events

**Type:** enhancement

winning 1.x could build a contestant's performance density from a *current score*
plus a list of distributions of *future scoring events* (`densities_from_events`,
`state_prices_from_events` in `lattice.py`) — the natural model for in-play golf,
cumulative-points tournaments, or any contest decided by a sum of increments. The
demo timed 125 contestants x 18 future events comfortably.

thurstone's `Density.convolve` already does the heavy lifting; what's missing is the
convenience constructor:

```python
Density.from_events(lattice, current=score_i, events=[d1, d2, ...])   # convolve chain
```

plus a shifted-anchor convention so current scores land on the lattice. Suggest it as
a small addition to `density.py` with one example script.

Reference implementation: `densities_from_events` in winning 1.x `lattice.py`
(git history), demo preserved at `attic/examples_events/` in the winning repo.
