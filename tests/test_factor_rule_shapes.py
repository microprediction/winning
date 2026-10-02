"""#75: a caller-supplied factor rule (F, W) is validated in _setup, before
any backend dispatch, so the compiled and pure paths raise the same
ValueError naming the mismatch instead of a PanicException / late matmul
error."""
import numpy as np
import pytest

import winning
from winning.factor import race_probabilities

MU = np.array([-0.2, 0.0, 0.2])
V1 = np.array([0.1, 0.2, 0.3])
D = np.ones(3)
F3 = np.array([[-1.0], [0.0], [1.0]])
W3 = np.array([0.25, 0.5, 0.25])


@pytest.fixture(params=[True, False], ids=["rust", "pure"])
def backend(request):
    winning.use_rust(request.param)
    yield request.param
    winning.use_rust(True)


@pytest.mark.parametrize("F,W,match", [
    (F3, np.array([0.5, 0.5]), "3 nodes but W has 2"),
    (F3, np.full(4, 0.25), "3 nodes but W has 4"),
    (np.zeros((3, 2)), W3, "2 factor columns but V has rank 1"),
    (np.zeros(3), W3, "F must be a 2-D"),
    (F3, W3[:, None], "W must be a 1-D"),
])
def test_mismatched_rule_is_value_error(backend, F, W, match):
    with pytest.raises(ValueError, match=match):
        race_probabilities(MU, V=V1, D=D, F=F, W=W, points=129)


@pytest.mark.parametrize("which", ["F", "W"])
def test_lone_rule_argument_is_refused(backend, which):
    kw = {"F": F3} if which == "F" else {"W": W3}
    with pytest.raises(ValueError, match="supply both F"):
        race_probabilities(MU, V=V1, D=D, points=129, **kw)


def test_valid_rule_still_runs(backend):
    p = race_probabilities(MU, V=V1, D=D, F=F3, W=W3, points=129)
    assert abs(p.sum() - 1) < 1e-9
