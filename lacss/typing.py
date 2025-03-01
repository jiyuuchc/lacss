from __future__ import annotations

import pathlib
from typing import Any, Mapping, Sequence, Union

import optax
from jax import Array
from jax.typing import ArrayLike

Optimizer = optax.GradientTransformation

PathLike = Union[str, pathlib.Path]

DataDict = Mapping[str, ArrayLike]

Patches = DataDict
