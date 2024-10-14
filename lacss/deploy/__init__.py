from typing import Mapping
from .predict import Predictor

model_urls: Mapping[str, str] = {
    "lacss3-small": "https://huggingface.co/jiyuuchc/lacss3-small/resolve/main/lacss3-small",
    "lacss3-base": "https://huggingface.co/jiyuuchc/lacss3-base/resolve/main/lacss3-base",
    "lacss3-small-cellpose": "https://huggingface.co/jiyuuchc/lacss3-small-cellpose/resolve/main/lacss3-small-c",
    "lacss3-small-livecell": "https://huggingface.co/jiyuuchc/lacss3-small-livecell/resolve/main/lacss3-small-l",
    "lacss3-small-nips": "https://huggingface.co/jiyuuchc/lacss3-small-nips/resolve/main/lacss3-small-n",
}

model_urls["default"] = model_urls["lacss3-base"]

