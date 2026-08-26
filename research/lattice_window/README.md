# The winner-bulk window: 33 lattice points

Peter's conjecture: hopeless runners waste lattice points, and sizing the
lattice by the horses that matter should save many. Measured: it saves more
than many, and the reason is sharper than the conjecture.

    field                     window width        points for ~1e-10 TV
    30 live + 70 hopeless     25.8 -> 7.1 (3.6x)  full: >129 (floors 6e-11), bulk: 33
    30 live + 470 hopeless    27.6 -> 7.3 (3.8x)  full: >129,                bulk: 33
    all live (control)        20.6 -> 6.2 (3.3x)  full: 129,                 bulk: 33

- The right window was never min-to-max of abilities; it is the bulk of the
  WINNER distribution G(x) = 1 - prod_j S_j(x), found by bisection with an
  exact omitted-mass bound 2*delta. Even all-live fields waste 3x width.
- The full window's accuracy FLOORS (~6e-11) at its own +-8 sd truncation;
  the bulk window's error is user-set by delta and keeps descending.
- Narrowing sacrifices nobody: a hopeless runner only wins by running a
  winner-class time, so its own win integrand lives in the bulk too.
- 33 bulk points beat 513 full-window points. For calibration loops
  (inversion iterates forwards) this is ~4x per iteration before the
  importance-tiered savings (freeze/mutuel-field the tail, Newton on the
  top-k block with the drop-bound <= dropped mass) are even applied.

Same construction as research/qpo's adaptive window (65 points = 4097 there);
this note isolates and quantifies the win on racing-shaped fields.
`run_window_savings.py` reproduces the table. Next: port into
winning/factor/races.py as the default window, delta a keyword.
