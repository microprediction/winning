from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UniformLattice:
    L: int  # half-width (number of steps on one side)
    unit: float  # spacing between lattice points

    @property
    def size(self) -> int:
        return 2 * self.L + 1

    @property
    def grid(self) -> np.ndarray:
        return self.unit * np.linspace(-self.L, self.L, self.size)

    def index_grid(self) -> np.ndarray:
        return np.arange(-self.L, self.L + 1, dtype=int)

    def assert_compatible(self, arr: np.ndarray) -> None:
        if arr.ndim != 1 or arr.shape[0] != self.size:
            raise ValueError(
                f"Array length {arr.shape[0]} incompatible with lattice size {self.size}."
            )
