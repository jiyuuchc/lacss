import logging
import threading
import traceback
from concurrent import futures

import grpc
import numpy as np
import typer
from cellpose import models

from . import proto
from .common import TokenValidationInterceptor, decode_image

app = typer.Typer(pretty_exceptions_enable=False)

_MAX_MSG_SIZE = 1024 * 1024 * 128
_TARGET_CELL_SIZE = 30


def process_input(request: proto.DetectionRequest):
    pixels = request.image_data.pixels
    settings = request.detection_settings

    image = decode_image(pixels)

    physical_size = pixels.physical_size_x or 1

    if settings.HasField("cell_diameter_hint"):
        diameter = settings.cell_diameter_hint / physical_size

    else:
        diameter = _TARGET_CELL_SIZE / (settings.scaling_hint or 1.0)

    if image.shape[0] == 1:  # 2D
        image = image.squeeze(0)

    if image.shape[-1] > 1:
        channels = [1, 2]
    else:
        channels = [0, 0]

    kwargs = dict(
        diameter=diameter,
        channels=channels,
    )

    return image, kwargs


def process_result(preds, image):
    import cv2
    from skimage.measure import regionprops

    response = proto.DetectionResponse()

    try:
        masks, flows, styles, _ = preds
    except:
        masks, flows, styles = preds

    for rp in regionprops(masks):
        mask = rp.image.astype("uint8")
        c, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = np.array(c[0], dtype=float).squeeze(1)
        c = c + np.array([rp.bbox[1], rp.bbox[0]])
        c = c - 0.5

        scored_roi = proto.ScoredROI(
            score=1.0,
            roi=proto.ROI(
                polygon=proto.Polygon(points=[proto.Point(x=p[0], y=p[1]) for p in c]),
            ),
        )

        response.detections.append(scored_roi)

    return response


class CellposeServicer(proto.LacssServicer):
    def __init__(self, model):
        self.model = model
        self._lock = threading.RLock()

    def RunDetection(self, request, context):
        with self._lock:

            logging.info(f"Received message of size {request.ByteSize()}")

            try:
                image, kwargs = process_input(request)

                if image.ndim == 4:
                    raise ValueError("cellpose_server does not support 3D input")

                logging.info(f"received image {image.shape}")

                preds = self.model.eval(
                    image,
                    **kwargs,
                )

                response = process_result(preds, image)

                logging.info(f"Reply with message of size {response.ByteSize()}")

                return response

            except ValueError as e:

                logging.error(repr(e))

                logging.error(traceback.format_exc())

                context.abort(grpc.StatusCode.INVALID_ARGUMENT, repr(e))

            except Exception as e:

                logging.error(repr(e))

                logging.error(traceback.format_exc())

                context.abort(
                    grpc.StatusCode.UNKNOWN, f"prediction failed with error: {repr(e)}"
                )

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


@app.command()
def main(
    modeltype: str = "cyto3",
    port: int = 50051,
    workers: int = 10,
    ip: str = "0.0.0.0",
    local: bool = False,
    token: bool | None = None,
    debug: bool = False,
    compression: bool = True,
    gpu: bool = True,
    # max_image_size: int = 1088,
):
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)

    print("server starting ...")

    model = models.Cellpose(model_type=modeltype, gpu=gpu)

    if token is None:
        token = not local
    if token:
        import secrets

        token_str = secrets.token_urlsafe(64)

        print()
        print("COPY THE TOKEN BELOW FOR ACCESS.")
        print("=======================================================================")
        print(f"{token_str}")
        print("=======================================================================")
        print()
    else:
        token_str = None

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=workers),
        compression=grpc.Compression.Gzip
        if compression
        else grpc.Compression.NoCompression,
        interceptors=(TokenValidationInterceptor(token_str),),
        options=(("grpc.max_receive_message_length", _MAX_MSG_SIZE),),
    )

    proto.add_LacssServicer_to_server(
        CellposeServicer(model),
        server,
    )

    if local:
        server.add_secure_port(f"127.0.0.1:{port}", grpc.local_server_credentials())
    else:
        server.add_insecure_port(f"{ip}:{port}")

    logging.info(f"lacss_server: listening on port {port}")

    print("server starting ... ready")

    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    app()
