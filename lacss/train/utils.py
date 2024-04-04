import dataclasses
import re
from collections import deque
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
    if isinstance(obj, str):
        return obj.split("/")[-1]
    elif hasattr(obj, "name") and obj.name:
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


def pack_x_y_sample_weight(x, y=None, sample_weight=None):
    """Packs user-provided data into a tuple."""
    if y is None:
        return (x,)
    elif sample_weight is None:
        return (x, y)
    else:
        return (x, y, sample_weight)


def unpack_x_y_sample_weight(data):
    """Unpacks user-provided data tuple."""
    if not isinstance(data, tuple):
        return (data, None, None)
    elif len(data) == 1:
        return (data[0], None, None)
    elif len(data) == 2:
        return (data[0], data[1], None)
    elif len(data) == 3:
        return (data[0], data[1], data[2])

    raise ValueError("Data not understood.")


def unpack_prediction_and_state(pred, mutable=None):
    if mutable is not None:
        if mutable:
            return pred[0], pred[1]
        else:
            return pred, {}
    if isinstance(pred, tuple):
        return pred[0], pred[1]
    else:
        return pred[0], {}


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
    from torch.utils.data import DataLoader

    class TorchDataLoaderAdapter:
        """Convert torch dataloader into a python iterable suitable for [lacss.Trainer](./#lacss.train.base_trainer.Trainer)

        ```
        my_dataset = TorchDataLoaderAdapter(my_torch_dataloader)
        ```

        """

        def __init__(
            self,
            x: DataLoader,
        ):
            self._dataset = x

        def get_dataset(self):
            def parse_dataloader_gen():
                for batch in iter(self._dataset):
                    batch = jax.tree_util.tree_map(
                        lambda x: x.cpu().numpy(), list_to_tuple(batch)
                    )
                    yield batch

            return parse_dataloader_gen()

        def __iter__(self):
            return self.get_dataset()

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

        def __init__(self, ds: Dataset):
            self._dataset = ds

        def __iter__(self):
            return self._dataset.as_numpy_iterator()

except ImportError:
    TFDatasetAdapter = None


class GeneratorAdapter:
    def __init__(self, g):
        self._generator = g

    def __iter__(self):
        return self._generator()


class Peekable:
    def __init__(self, iterator):
        self.iterator = iterator
        self.peeked = deque()

    def __iter__(self):
        return self.iterator

    def __next__(self):
        if self.peeked:
            return self.peeked.popleft()
        return next(self.iterator)

    def peek(self, ahead=0):
        while len(self.peeked) <= ahead:
            self.peeked.append(next(self.iterator))
        return self.peeked[ahead]
