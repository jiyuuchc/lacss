import threading

import biopb.image as proto
import grpc
import numpy as np
from biopb.image.utils import deserialize_to_numpy

_AUTH_HEADER_KEY = "authorization"


def decode_image(pixels: proto.Pixels) -> np.ndarray:
    if pixels.size_t > 1:
        raise ValueError("Image data has a non-singleton T dimension.")

    if pixels.size_c > 3:
        raise ValueError("Image data has more than 3 channels.")

    np_img = deserialize_to_numpy(pixels)

    return np_img


class TokenValidationInterceptor(grpc.ServerInterceptor):
    def __init__(self, token):
        def abort(ignored_request, context):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid token signature")

        self._abort_handler = grpc.unary_unary_rpc_method_handler(abort)
        self.token = token

    def intercept_service(self, continuation, handler_call_details):
        expected_metadata = (_AUTH_HEADER_KEY, f"Bearer {self.token}")
        if (
            self.token is None
            or expected_metadata in handler_call_details.invocation_metadata
        ):
            return continuation(handler_call_details)
        else:
            return self._abort_handler


class LacssServicerBase(proto.ObjectDetectionServicer, proto.ProcessImageServicer):
    def __init__(self):
        self._lock = threading.RLock()

    def RunDetectionStream(self, request_iterator, context):
        with self._lock:
            request = proto.DetectionRequest()

            for next_request in request_iterator:

                if next_request.image_data.HasField("pixels"):
                    request.image_data.pixels.CopyFrom(next_request.image_data.pixels)

                if next_request.image_data.HasField("image_annotation"):
                    request.image_data.image_annotation.CopyFrom(
                        next_request.image_data.image_annotation
                    )

                if next_request.HasField("detection_settings"):
                    request.detection_settings.CopyFrom(next_request.detection_settings)

                if request.image_data.HasField("pixels"):
                    yield self.RunDetection(request, context)
