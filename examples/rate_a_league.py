"""Rate a small league from finish orders and predict the next race.

    python examples/rate_a_league.py
"""

from winning import ThurstoneRating

tr = ThurstoneRating()

results = [
    (["ada", "ben", "cid", "dot"], [1, 2, 3, 4]),
    (["ada", "cid", "eve"], [1, 2, 3]),
    (["ben", "cid", "dot", "eve"], [2, 1, 3, 4]),
    (["ada", "ben", "eve"], [1, 3, 2]),
    (["cid", "dot", "eve"], [1, 2, 3]),
]
for names, ranks in results:
    tr.observe(names, ranks)

print("Leaderboard (conservative):")
for name, rating in tr.leaderboard():
    print(f"  {name:5s} mu={rating.mu:+.3f} sigma={rating.sigma:.3f}")

field = ["ada", "cid", "eve"]
probs = tr.win_probabilities(field)
print("\nNext race win probabilities:")
for name, p in zip(field, probs):
    print(f"  {name:5s} {p:.3f}")
