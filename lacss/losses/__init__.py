from .auxiliary import *
from .instance import *
from .common import *

def lpn_loss(batch, prediction):
    return prediction["losses"]["lpn_detection_loss"] + prediction["losses"]["lpn_localization_loss"]
