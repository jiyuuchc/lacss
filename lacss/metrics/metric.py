import typing as tp
from abc import abstractmethod


class Metric:
    @abstractmethod
    def update(self, *args, **kwargs):
        ...

    @abstractmethod
    def compute(self, *args, **kwargs) -> tp.Dict:
        ...
