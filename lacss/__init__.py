import importlib.metadata

_LACSSS_DISTRIBUTION_METADATA = importlib.metadata.metadata("lacss")

__version__ = _LACSSS_DISTRIBUTION_METADATA["version"]
