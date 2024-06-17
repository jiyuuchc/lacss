from __future__ import annotations

import pathlib
import typing

import optax
from jax import Array
from jax.typing import ArrayLike

Optimizer = optax.GradientTransformation

PathLike = typing.Union[str, pathlib.Path]

class Patches(typing.TypedDict):
    instance_output: ArrayLike
    instance_yc: ArrayLike
    instance_xc: ArrayLike


DataDict = typing.Mapping[str, ArrayLike]
