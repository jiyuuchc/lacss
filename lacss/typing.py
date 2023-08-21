from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol, TypedDict, Union

import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from jax import Array
from jax.typing import ArrayLike

Params = FrozenDict

Optimizer = optax.GradientTransformation

PathLike = Union[str, Path]


class LossFunc(Protocol):
    def __call__(self, batch: Any, prediction: Any) -> float:
        ...


class Metric(Protocol):
    def update(self, batch: Any, prediction: Any):
        ...

    def compute(self, *args, **kwargs) -> dict:
        ...


class Patches(TypedDict):
    instance_output: ArrayLike
    instance_yc: ArrayLike
    instance_xc: ArrayLike


DataDict = Mapping[str, ArrayLike]

__all__ = [
    "Array",
    "ArrayLike",
    "DataDict",
    "Params",
    "Optimizer",
    "LossFunc",
    "Metric",
    "Patches",
    "PathLike",
]
