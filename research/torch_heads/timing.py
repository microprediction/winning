"""Timing and memory of winning.torch.race_log_prob: analytic vs autograd.

    OMP_NUM_THREADS=4 python research/torch_heads/timing.py

Forward + backward of sum(G * race_log_prob(mu, sd)) at points=257, for
batch x n in {(256, 10), (64, 100), (8, 1000)}. "tape" is the bytes autograd
saves for the backward (summed through saved_tensors_hooks): the analytic
Function saves only its inputs, outputs and the lattice; autograd through
the forward keeps every (batch, n, L) intermediate. Median of repeats.
"""
import os
import statistics
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")

import torch  # noqa: E402

import winning.torch as wt  # noqa: E402

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
SHAPES = [(256, 10), (64, 100), (8, 1000)]


def _tape_bytes(fn):
    total = [0]

    def pack(t):
        total[0] += t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return out, total[0]


def run(device, dtype, mode, B, n, reps=7):
    how, ck = {"analytic": ("analytic", False),
               "checkpoint": ("analytic", True),
               "autograd": ("autograd", False)}[mode]
    g = torch.Generator().manual_seed(0)
    mu0 = torch.randn(B, n, generator=g, dtype=dtype).to(device)
    sd0 = (0.5 + torch.rand(B, n, generator=g, dtype=dtype)).to(device)
    G = torch.randn(B, n, generator=g, dtype=dtype).to(device)

    def once():
        mu = mu0.clone().requires_grad_(True)
        sd = sd0.clone().requires_grad_(True)
        lp, tape = _tape_bytes(lambda: wt.race_log_prob(
            mu, sd, backward=how, checkpoint=ck))
        (lp * G).sum().backward()
        return tape

    tape = once()                               # warm-up
    ts = []
    for _ in range(reps):
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        once()
        if device == "mps":
            torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), tape


def main():
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}, "
          "points=257, forward+backward, median of 7")
    modes = ("analytic", "checkpoint", "autograd")
    print("ms = forward+backward; MB = autograd tape (saved tensors)")
    print(f"{'device':6s} {'dtype':8s} {'B x n':>11s} | "
          + " | ".join(f"{m:>10s} ms    MB" for m in modes)
          + " | autograd/analytic")
    for device in devices:
        for dtype in ([torch.float32, torch.float64] if device == "cpu"
                      else [torch.float32]):
            for B, n in SHAPES:
                row = {}
                for m in modes:
                    try:
                        row[m] = run(device, dtype, m, B, n)
                    except (RuntimeError, NotImplementedError,
                            TypeError) as exc:
                        print(f"  {device} {m} failed: "
                              f"{str(exc).splitlines()[0][:100]}")
                        row[m] = None
                if None in row.values():
                    continue
                cells = " | ".join(f"{1e3 * row[m][0]:13.2f} "
                                   f"{row[m][1] / 2**20:5.1f}" for m in modes)
                print(f"{device:6s} {str(dtype)[6:]:8s} {B:>4d} x {n:<5d}| "
                      f"{cells} | "
                      f"{row['autograd'][0] / row['analytic'][0]:5.2f}x")


if __name__ == "__main__":
    main()
