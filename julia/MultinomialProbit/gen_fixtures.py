"""Generate cross-language fixtures for MultinomialProbit.jl from the
python reference (winning.likelihood / winning.mnprobit are the spec).
Inputs AND outputs embedded so Julia consumes identical floats.

Run:  python julia/MultinomialProbit/gen_fixtures.py
"""
import json
import os

import numpy as np

from winning.likelihood import choice_loglik_and_score
from winning.mnprobit import MNProbit, _prob_of

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    rng = np.random.default_rng(2026)
    T, J, p, r = 40, 4, 2, 2
    X = np.round(rng.normal(size=(T, J, p)), 12)
    beta_true = np.array([0.8, -0.5])
    V_true = np.array([[0.0, 0.0], [0.6, 0.0], [-0.4, 0.5], [0.2, -0.3]])
    mu = X @ beta_true
    f = rng.normal(size=(T, r))
    z = rng.normal(size=(T, J))
    U = mu + f @ V_true.T + z
    choice = U.argmax(axis=1)

    cases = []
    for name, beta, V in (
        ("at_truth", beta_true, V_true),
        ("off_truth", np.array([0.3, 0.1]),
         np.array([[0.0, 0.0], [0.2, 0.0], [0.1, -0.2], [-0.3, 0.4]])),
    ):
        m = X @ beta
        ll, dmu, dV = choice_loglik_and_score(m, V, choice)
        P = np.column_stack([_prob_of(m[:5], V - V.mean(axis=0), k)
                             for k in range(J)])
        P = P / P.sum(axis=1, keepdims=True)
        cases.append({
            "name": name, "beta": beta.tolist(), "V": V.tolist(),
            "loglik": ll, "dmu": dmu.tolist(), "dV": dV.tolist(),
            "proba5": P.tolist(),
        })

    out = {
        "T": T, "J": J, "p": p, "r": r,
        "X": X.tolist(), "choice": choice.tolist(),   # 0-based
        "cases": cases,
    }
    path = os.path.join(HERE, "test", "vectors.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {path}: {len(cases)} cases, loglik[0] = "
          f"{cases[0]['loglik']:.6f}")


if __name__ == "__main__":
    main()
