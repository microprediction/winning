# TEMPORARY debug probe for the Linux-only NaN (removed before merge)
import warnings
import numpy as np
import winning.factor as wf
import winning.factor.races as R


def test_zz_debug_669_dump():
    out = []
    np.set_printoptions(precision=17)
    for rho in [0, 2]:
        Vl = [[0.5, -0.25], [0.25, 0.5], [-0.5, 0.125], [0.125, 0.25]]
        Vm = np.array(Vl)[:, :rho] if rho else np.zeros((4, 0))
        m, V1, D1, F1, W1, rep = wf.update_contrast(
            np.zeros(4), (Vm, np.array([1.0, 0.5, 2.0, 0.75])),
            np.array([0.5, 2, -1, 3.]), mean=1.0, var=1e-20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, V, D, F, W, fn, left, right = R._setup(-m, -V1, D1, None, None, "normal")
            M_all = mu[None, :] + F @ V.T
            sd = np.sqrt(D)
            x0 = R._bulk_window(M_all, sd, 257, 1e-12, fn)
            x, points = R.forward_grid(M_all, sd, V, fn, left, right, 257)
        out.append(f"rho={rho} m={m.tolist()!r}\nV1={V1.tolist()!r}\nD1={D1.tolist()!r}")
        out.append(f"bulk0 {x0[0]!r} {x0[-1]!r} {len(x0)}  grid {x[0]!r} {x[-1]!r} {points}")
        out.append(f"F[:3]={F[:3].tolist()!r} Msum={float(M_all.sum())!r}")
    raise AssertionError("DEBUG669\n" + "\n".join(out))
