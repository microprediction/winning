"""Run the rating-system benchmark and print a markdown results table.

    python -m winning.benchmarks.run_benchmark --dataset synthetic
    python -m winning.benchmarks.run_benchmark --list

Datasets are fetched at run time and cached under ~/.cache/winning (override
with WINNING_CACHE). Kaggle-hosted datasets need ~/.kaggle/kaggle.json.
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict, List, Tuple

from ..elo import EloRating
from ..glicko2 import Glicko2Rating
from ..ratingsystem import Rating, RatingSystem
from ..thurstonerating import ThurstoneRating
from .events import Event, synthetic_world
from .forward_chain import evaluate


class UniformBaseline(RatingSystem):
    def observe(self, names, ranks, dt=1.0):
        pass

    def win_probabilities(self, names):
        return [1.0 / len(names)] * len(names)

    def rating(self, name):
        return Rating(mu=0.0, sigma=None)

    def known(self):
        return []


def build_systems(
    include_third_party: bool = True, draw_probability: float = 0.0
) -> Dict[str, RatingSystem]:
    systems: Dict[str, RatingSystem] = {
        "Elo (multi-entrant)": EloRating(),
        "Glicko-2": Glicko2Rating(),
        "Thurstone lattice (winning)": ThurstoneRating(),
    }
    if include_third_party:
        try:
            from ..shims import OpenSkillRating, TrueSkillRating

            systems["TrueSkill"] = TrueSkillRating(draw_probability=draw_probability)
            systems["OpenSkill PlackettLuce"] = OpenSkillRating("PlackettLuce")
            systems["OpenSkill ThurstoneMostellerFull"] = OpenSkillRating(
                "ThurstoneMostellerFull"
            )
            systems["OpenSkill PlackettLuce (lattice predict)"] = OpenSkillRating(
                "PlackettLuce", exact_predict=True
            )
        except ImportError:
            print("(third-party comparators missing; pip install winning[benchmarks])")
    return systems


def run(events: List[Event]) -> List[dict]:
    """Evaluate every system, plus baseline rows for whatever the events carry:
    a Uniform floor always, an Oracle row when events have truth, and a Market
    row when events have market-implied probabilities."""
    rows = []
    # TrueSkill needs the draw rate declared up front; give it the empirical one
    n_tied = sum(1 for ev in events if sorted(ev.ranks).count(min(ev.ranks)) > 1)
    systems = build_systems(draw_probability=n_tied / max(1, len(events)))
    systems["Uniform baseline"] = UniformBaseline()
    # rank-PIT needs a finish-position distribution, which is trivial at N=2
    rank_pit = any(len(ev.names) > 2 for ev in events)

    for label, system in systems.items():
        m = evaluate(system, events, rank_pit=rank_pit)
        row = {"system": label}
        row.update(m.summary())
        rows.append(row)
        print(f"  done: {label}  log_loss={row['log_loss']:.4f}")

    for attr, label in (("truth", "Oracle (true abilities)"), ("market", "Market odds")):
        if any(getattr(ev, attr) is not None for ev in events):
            m = evaluate(None, events, fixed=attr)
            row = {"system": label}
            row.update(m.summary())
            rows.append(row)
            print(f"  done: {label}  log_loss={row['log_loss']:.4f}")
    return rows


def to_markdown(rows: List[dict], title: str) -> str:
    cols = [
        "system",
        "log_loss",
        "brier",
        "accuracy",
        "kendall_tau",
        "ece",
        "rank_pit_ks",
        "tv_vs_oracle",
        "seconds",
    ]
    heads = [
        "System",
        "Log loss",
        "Brier",
        "Accuracy",
        "Kendall tau",
        "ECE",
        "Rank-PIT KS",
        "TV vs oracle",
        "Seconds",
    ]
    lines = [f"### {title}", "", "| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for row in sorted(rows, key=lambda r: r["log_loss"]):
        cells = []
        for c in cols:
            v = row.get(c)
            if v is None:
                cells.append("-")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}" if c != "seconds" else f"{v:.1f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------- dataset registry ----------------


def _synthetic(args) -> Tuple[List[Event], str]:
    events = synthetic_world(num_contestants=args.contestants, num_events=args.events)
    return events, (
        f"Synthetic races: {args.contestants} contestants, {args.events} events, "
        "fields of 6-12, static abilities"
    )


def _synthetic_drift(args) -> Tuple[List[Event], str]:
    events = synthetic_world(
        num_contestants=args.contestants, num_events=args.events, drift_tau=0.01
    )
    return events, (
        f"Synthetic races with ability drift (tau=0.01/event): "
        f"{args.contestants} contestants, {args.events} events"
    )


def _tennis(tour: str):
    def build(args) -> Tuple[List[Event], str]:
        from .tennis import tennis_events

        events = tennis_events(tour=tour)
        return events, (
            f"{tour.upper()} tennis 2013-2024 ({len(events)} matches; Sackmann archive)"
        )

    return build


def _synthetic_hetero(args) -> Tuple[List[Event], str]:
    events = synthetic_world(
        num_contestants=args.contestants, num_events=args.events, noise_sigmas=(0.6, 1.6)
    )
    return events, (
        f"Synthetic races, heteroskedastic noise (per-contestant sd 0.6 or 1.6): "
        f"{args.contestants} contestants, {args.events} events"
    )


DATASETS: Dict[str, Callable] = {
    "synthetic": _synthetic,
    "synthetic-drift": _synthetic_drift,
    "synthetic-hetero": _synthetic_hetero,
    "atp": _tennis("atp"),
    "wta": _tennis("wta"),
    "tennis": _tennis("atp"),  # historical alias
}


def register(name: str, builder: Callable) -> None:
    DATASETS[name] = builder


def _register_optional() -> None:
    """Loaders whose modules exist but may be gated on external availability."""
    try:
        from . import f1  # noqa: F401

        DATASETS["f1"] = lambda args: f1.dataset(args)
    except ImportError:
        pass
    try:
        from . import chess  # noqa: F401

        DATASETS["chess"] = lambda args: chess.dataset(args)
    except ImportError:
        pass
    try:
        from . import football  # noqa: F401

        DATASETS["epl"] = lambda args: football.dataset(args)
    except ImportError:
        pass
    try:
        from . import sumo  # noqa: F401

        DATASETS["sumo"] = lambda args: sumo.dataset(args)
    except ImportError:
        pass
    try:
        from . import halo2  # noqa: F401

        DATASETS["halo2"] = lambda args: halo2.dataset(args)
        DATASETS["halo2-ffa"] = lambda args: halo2.dataset_ffa(args)
    except ImportError:
        pass
    try:
        from . import kaggle_datasets  # noqa: F401

        DATASETS["hkracing"] = lambda args: kaggle_datasets.hkracing(args)
        DATASETS["pubg"] = lambda args: kaggle_datasets.pubg(args)
    except ImportError:
        pass


def main():
    _register_optional()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic", choices=sorted(DATASETS))
    ap.add_argument("--list", action="store_true", help="list datasets and exit")
    ap.add_argument("--events", type=int, default=4000)
    ap.add_argument("--contestants", type=int, default=200)
    ap.add_argument("--out", default=None, help="append markdown results to this file")
    args = ap.parse_args()

    if args.list:
        for name in sorted(DATASETS):
            print(name)
        return

    events, title = DATASETS[args.dataset](args)
    print(f"Running: {title}")
    rows = run(events)
    md = to_markdown(rows, title)
    print()
    print(md)
    if args.out:
        with open(args.out, "a") as f:
            f.write(md + "\n\n")


if __name__ == "__main__":
    main()
