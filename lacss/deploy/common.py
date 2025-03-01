import threading

import grpc
import numpy as np

from . import proto

_AUTH_HEADER_KEY = "authorization"


def get_dtype(pixels: proto.Pixels) -> np.dtype:
    dt = np.dtype(pixels.dtype)

    if pixels.bindata.endianness == proto.BinData.Endianness.BIG:
        dt = dt.newbyteorder(">")
    else:
        dt = dt.newbyteorder("<")

    return dt


def decode_image(pixels: proto.Pixels) -> np.ndarray:
    if pixels.size_t > 1:
        raise ValueError("Image data has a non-singleton T dimension.")

    if pixels.size_c > 3:
        raise ValueError("Image data has more than 3 channels.")

    np_img = np.frombuffer(
        pixels.bindata.data,
        dtype=get_dtype(pixels),
    ).astype("float32")

    # The dimension_order describe axis order but in the F_order convention
    # Numpy default is C_order, so we reverse the sequence. Lacss expect the
    # final dimension order to be "ZYXC"
    dim_order_c = pixels.dimension_order[::-1].upper()
    dims = dict(
        Z=pixels.size_z or 1,
        Y=pixels.size_y or 1,
        X=pixels.size_x or 1,
        C=pixels.size_c or 1,
        T=1,
    )
    dim_orig = [dim_order_c.find(k) for k in "ZYXCT"]
    shape_orig = [dims[k] for k in dim_order_c]

    np_img = np_img.reshape(shape_orig).transpose(dim_orig)

    np_img = np_img.squeeze(axis=-1)  # remove T

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


class LacssServicerBase(proto.LacssServicer):
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
