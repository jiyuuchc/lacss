from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from jax import Array

try:
    from jax.typing import ArrayLike
except:
    ArrayLike = Union[
        Array,  # JAX array type
        np.ndarray,  # NumPy array type
        np.bool_,
        np.number,  # NumPy scalar types
        bool,
        int,
        float,
        complex,  # Python scalar types
    ]


DataDict = Mapping[str, Array]

Params = FrozenDict

DType = np.dtype

Shape = Sequence[int]

Optimizer = optax.GradientTransformation

PathLike = Union[str, Path]

DataSource = Union[Iterator, callable]


class LossFunc(Protocol):
    def __call__(self, preds: DataDict, labels: DataDict, inputs: Any) -> ArrayLike:
        ...


class Patches(TypedDict):
    instance_output: ArrayLike
    instance_yc: ArrayLike
    instance_xc: ArrayLike
