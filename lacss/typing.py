from __future__ import annotations

import pathlib
import typing

import optax
from jax import Array
from jax.typing import ArrayLike

Optimizer = optax.GradientTransformation

PathLike = typing.Union[str, pathlib.Path]

DataDict = typing.Mapping[str, ArrayLike]

Patches = DataDict

