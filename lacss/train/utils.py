import dataclasses
import re
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import jax
from flax import struct

from ..typing import *

InputLike = Any


def _unique_name(
    names: Set[str],
    name: str,
):

    if name in names:

        match = re.match(r"(.*?)(\d*)$", name)
        assert match is not None

        name = match[1]
        num_part = match[2]

        i = int(num_part) if num_part else 2
        str_template = f"{{name}}{{i:0{len(num_part)}}}"

        while str_template.format(name=name, i=i) in names:
            i += 1

        name = str_template.format(name=name, i=i)

    names.add(name)
    return name


def _unique_names(
    names: Iterable[str],
    *,
    existing_names: Optional[Set[str]] = None,
) -> Iterable[str]:
    if existing_names is None:
        existing_names = set()

    for name in names:
        yield _unique_name(existing_names, name)


def _lower_snake_case(s: str) -> str:
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
    parts = s.split("_")
    output_parts = []

    for i in range(len(parts)):
        if i == 0 or len(parts[i - 1]) > 1:
            output_parts.append(parts[i])
        else:
            output_parts[-1] += parts[i]

    return "_".join(output_parts)


def _get_name(obj) -> str:
    if hasattr(obj, "name") and obj.name:
        return obj.name
    elif hasattr(obj, "func") and obj.func.__name__:
        return _lower_snake_case(obj.func.__name__)
    elif hasattr(obj, "__name__") and obj.__name__:
        return _lower_snake_case(obj.__name__)
    elif hasattr(obj, "__class__") and obj.__class__.__name__:
        return _lower_snake_case(obj.__class__.__name__)
    else:
        raise ValueError(f"Could not get name for: {obj}")


def list_to_tuple(maybe_list):
    if isinstance(maybe_list, list):
        return tuple(maybe_list)
    return maybe_list


class Inputs(struct.PyTreeNode):
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def update(self, *args, **kwargs):
        tmp = self.kwargs.copy()
        tmp.update(kwargs)
        new_inputs = self.replace(args=self.args + args, kwargs=tmp)
        return new_inputs

    @classmethod
    def from_value(cls, value: InputLike) -> "Inputs":
        if isinstance(value, cls):
            return value
        elif isinstance(value, tuple):
            return cls(args=value)
        elif isinstance(value, dict):
            return cls(kwargs=value)
        else:
            return cls(args=(value,))


try:
    from .torch_dataloader_adapter import TorchDataLoaderAdapter

    class TorchDataLoaderAdapter:
        """Convert torch dataloader into a python iterable suitable for [lacss.Trainer](./#lacss.train.base_trainer.Trainer)

        ```
        my_dataset = TorchDataLoaderAdapter(my_torch_dataloader)
        ```

        """

        def __init__(
            self,
            x: DataLoader,
            steps: int = -1,
        ):
            self.steps = steps
            self._dataset = x
            self.current_step = 0

        def get_dataset(self):
            def parse_dataloader_gen():
                self.current_step = 0
                for batch in iter(self._dataset):
                    self.current_step += 1
                    batch = jax.tree_util.tree_map(
                        lambda x: x.cpu().numpy(), list_to_tuple(batch)
                    )
                    yield batch

            return parse_dataloader_gen

        def __iter__(self):
            g = self.get_dataset()
            return g()

except ImportError:
    TorchDataLoaderAdapter = None


try:
    from tensorflow.data import Dataset

    class TFDatasetAdapter:
        """Convert `tf.data.Dataset` into a python iterable suitable for [lacss.train.Trainer](./#lacss.train.base_trainerxtrain.Trainer)

        ```
        my_dataset = TFDatasetAdapter(my_tf_dataset)
        ```

        """

        def __init__(self, ds: Dataset, steps=-1):
            self._dataset = ds

        def get_dataset(self):
            def parse_tf_data_gen():
                for batch in iter(self._dataset):
                    batch = jax.tree_util.tree_map(lambda x: x.numpy(), batch)
                    yield batch

            return parse_tf_data_gen

        def __iter__(self):
            g = self.get_dataset()
            return g()

except ImportError:
    TFDatasetAdaptor = None
