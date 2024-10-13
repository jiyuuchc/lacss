from typing import Mapping
from .predict import Predictor

model_urls: Mapping[str, str] = {
    "lacss_v2_default": "https://huggingface.co/yulabuchc/lacss_v2_default/resolve/main/LACSSV2-DEFAULT-240927"
}
model_urls["default"] = model_urls["lacss_v2_default"]

