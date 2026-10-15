"""A Thurstone head vs a softmax head, trained on the same choices.

    OMP_NUM_THREADS=4 python research/torch_heads/demo.py

Shows the training loop works; it is NOT a paper claim (one seed, one
synthetic world, a linear head).

World: items carry d features; the performance is X = x . w* + N(0, 1),
min-wins (a Thurstone case V model). Training data: menus of 4, the
winner only. Both heads are linear, mu = x . w, fitted by Adam on the
winner NLL -- one with winning.torch.thurstone_nll, one with
cross-entropy on -mu (Luce / softmax). Evaluation is under RESTRICTION,
not on held-out loss alone (Peter's plan): held-out menus of 4, larger
menus of 12 (never trained on), and second choices on menus of 4 (the
runner-up given the winner; Thurstone uses the ordered-prefix
probability, softmax its Plackett-Luce stage). Lower NLL is better; the
true model's NLL is the floor.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch  # noqa: E402

import winning.torch as wt  # noqa: E402

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
DT = torch.float64


def world(n_menus, menu, d, w_star, gen):
    x = torch.randn(n_menus, menu, d, generator=gen, dtype=DT)
    perf = x @ w_star + torch.randn(n_menus, menu, generator=gen, dtype=DT)
    order = torch.argsort(perf, dim=-1)                      # min-wins
    return x, order[:, 0], order[:, 1]


def nll(head, w, x, first, second=None):
    """Mean NLL of the winner, or of the runner-up given the winner."""
    mu = x @ w
    idx = torch.arange(len(x))
    if head == "softmax":
        lp = torch.log_softmax(-mu, dim=-1)
        if second is None:
            return -lp[idx, first].mean()
        mask = torch.ones_like(mu, dtype=torch.bool)
        mask[idx, first] = False
        lp2 = torch.log_softmax(torch.where(mask, -mu, -torch.inf), dim=-1)
        return -lp2[idx, second].mean()
    lp = wt.race_log_prob(mu)
    if second is None:
        return -lp[idx, first].mean()
    pre = wt.ordered_prefix_log_prob(mu, torch.stack([first, second], -1))
    return -(pre - lp[idx, first]).mean()


def fit(head, x, first, d, steps=300, lr=0.05):
    w = torch.zeros(d, dtype=DT, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = nll(head, w, x, first)
        loss.backward()
        opt.step()
    return w.detach()


def main(seed=0, d=5):
    gen = torch.Generator().manual_seed(seed)
    w_star = torch.randn(d, generator=gen, dtype=DT)
    x_tr, f_tr, _ = world(4000, 4, d, w_star, gen)
    x_te, f_te, s_te = world(4000, 4, d, w_star, gen)
    x_big, f_big, _ = world(2000, 12, d, w_star, gen)
    ws = {"truth (Thurstone, w*)": ("thurstone", w_star)}
    for head in ("thurstone", "softmax"):
        ws[f"{head} head"] = (head, fit(head, x_tr, f_tr, d))
    print(f"seed {seed}: trained on 4000 winners from menus of 4\n")
    print(f"{'model':24s} {'menus of 4':>11s} {'menus of 12':>12s} "
          f"{'2nd choice':>11s}  |w| / |w*|")
    for name, (head, w) in ws.items():
        with torch.no_grad():
            a = nll(head, w, x_te, f_te)
            b = nll(head, w, x_big, f_big)
            c = nll(head, w, x_te, f_te, s_te)
        print(f"{name:24s} {a:11.4f} {b:12.4f} {c:11.4f}  "
              f"{float(w.norm() / w_star.norm()):.3f}")


if __name__ == "__main__":
    main()
