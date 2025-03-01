import logging
import threading
import traceback
from concurrent import futures
from pathlib import Path
from typing import Iterable

import grpc
import jax
import numpy as np
import typer

from . import proto
from .common import (
    LacssServicerBase,
    TokenValidationInterceptor,
    decode_image,
    get_dtype,
)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s'
)

logger = logging.getLogger(__name__)

app = typer.Typer(pretty_exceptions_enable=False)

_MAX_MSG_SIZE = 1024 * 1024 * 128
_TARGET_CELL_SIZE = 32
_MAX_STREAM_MSG_SIZE = _MAX_MSG_SIZE * 16
MAX_IMG_SIZE = 1024 * 1024 * 1024


def _process_input(request: proto.DetectionRequest, image=None):
    pixels = request.image_data.pixels
    settings = request.detection_settings

    if image is None:
        image = decode_image(pixels)

    physical_size = np.array(
        [
            pixels.physical_size_z
            or pixels.physical_size_x,  # one might set xy but not z
            pixels.physical_size_y,
            pixels.physical_size_x,
        ],
        dtype="float",
    )
    if (physical_size == 0).any():
        physical_size[:] = 1.0

    if settings.HasField("cell_diameter_hint"):
        scaling = _TARGET_CELL_SIZE / settings.cell_diameter_hint * physical_size

    else:
        if physical_size[1] != physical_size[2]:
            raise ValueError("Scaling hint provided, but pixel is not isometric")

        scaling = np.array([settings.scaling_hint or 1.0] * 3, dtype="float")
        scaling[0] *= physical_size[0] / physical_size[1]
    
    logger.info(f"Requested rescaling factor is {scaling}")

    shape_hint = tuple(np.round(scaling * image.shape[:3]).astype(int))

    if image.shape[0] == 1:  # 2D
        image = image.squeeze(0)
        shape_hint = shape_hint[1:]

    kwargs = dict(
        reshape_to=shape_hint,
        score_threshold=settings.min_score or 0.4,
        min_area=settings.min_cell_area,
        nms_iou=settings.nms_iou or 0.4,
        segmentation_threshold=settings.segmentation_threshold or 0.5,
    )

    return image, kwargs


def _process_result(preds, image) -> proto.DetectionResponse:
    response = proto.DetectionResponse()

    if image.ndim == 3:  # returns polygon

        for contour, score in zip(preds["pred_contours"], preds["pred_scores"]):
            if len(contour) == 0:
                continue

            scored_roi = proto.ScoredROI(
                score=score,
                roi=proto.ROI(
                    polygon=proto.Polygon(
                        points=[proto.Point(x=p[0], y=p[1]) for p in contour]
                    ),
                ),
            )

            response.detections.append(scored_roi)

    else:  # 3d returns Mesh
        for mesh, score in zip(preds["pred_contours"], preds["pred_scores"]):
            scored_roi = proto.ScoredROI(
                score=score,
                roi=proto.ROI(
                    mesh=proto.Mesh(
                        verts=[
                            proto.Point(z=v[0], y=v[1], x=v[2]) for v in mesh["verts"]
                        ],
                        faces=[
                            proto.Mesh.Face(p1=p[0], p2=p[1], p3=p[2])
                            for p in mesh["faces"]
                        ],
                    ),
                ),
            )

            response.detections.append(scored_roi)

    return response


def _process_grid_input(request_iterator: Iterable[proto.DetectionRequest]):
    d, h, w, c = 0, 0, 0, 0
    images, grids = [], []
    request = None
    total_msg_size = 0
    for request in request_iterator:
        total_msg_size += request.ByteSize()
        if total_msg_size > _MAX_STREAM_MSG_SIZE:
            raise ValueError("input message size  {total_msg_size} exceeded limit.")

        pixels = request.image_data.pixels

        image = decode_image(pixels)

        grids.append(
            [
                pixels.offset_z,
                pixels.offset_y,
                pixels.offset_x,
            ]
        )
        images.append(image)

        d = max(d, pixels.offset_z + image.shape[0])
        h = max(d, pixels.offset_z + image.shape[1])
        w = max(d, pixels.offset_z + image.shape[2])
        c = max(c, image.shape[3])

        if d * h * w > MAX_IMG_SIZE:
            raise ValueError(f"input image is too large {(d, h, w)}")

    # empty input iterator
    if request is None:
        return None, None

    assert c <= 3

    full_image = np.zeros([d, h, w, c], dtype=images[0].dtype)
    for image, grid in zip(images, grids):
        full_image[
            grid[0] : grid[0] + image.shape[0],
            grid[1] : grid[1] + image.shape[1],
            grid[2] : grid[2] + image.shape[2],
            : image.shape[3],
        ] = image

    return _process_input(request, full_image)


class LacssServicer(LacssServicerBase):
    def __init__(self, model):
        super().__init__()

        self.model = model

    def RunDetection(self, request, context):
        logger.info(f"Received message of size {request.ByteSize()}")

        try:
            image, kwargs = _process_input(request)

            logger.info(f"received image {image.shape}")

        except Exception as e:
            logger.error(repr(e))

            logger.error(traceback.format_exc())

            context.abort(grpc.StatusCode.INVALID_ARGUMENT, repr(e))

            return

        try:
            with self._lock:
                preds = self.model.predict(
                    image,
                    output_type="contour",
                    **kwargs,
                )

            response = _process_result(preds, image)

            logger.info(f"Reply with message of size {response.ByteSize()}")

            return response

        except Exception as e:

            logger.error(repr(e))

            logger.error(traceback.format_exc())

            context.abort(
                grpc.StatusCode.UNKNOWN, f"prediction failed with error: {repr(e)}"
            )

    def RunDetectionOnGrid(self, request_iterator, context):
        try:
            image, kwargs = _process_grid_input(request_iterator)

            if image is None:
                return proto.DetectionResponse()

            logger.info(f"Received full image of size {image.shape[:-1]}")

            with self._lock:
                preds = self.model.predict(
                    image,
                    output_type="contour",
                    **kwargs,
                )

            response = _process_result(preds, image)

            logger.info(f"Reply with message of size {response.ByteSize()}")

            return response

        except ValueError as e:
            logger.error(repr(e))

            logger.error(traceback.format_exc())

            context.abort(grpc.StatusCode.INVALID_ARGUMENT, repr(e))

        except Exception as e:

            logger.error(repr(e))

            logger.error(traceback.format_exc())

            context.abort(
                grpc.StatusCode.UNKNOWN, f"prediction failed with error: {repr(e)}"
            )


def get_predictor(modelpath, f16):
    from .predict import Predictor

    model = Predictor(
        modelpath, 
        f16=f16,
        # grid_size=1088,
        # step_size=1024,
        # cells_per_grid=1024,
        # grid_size_3d=256,
        # step_size_3d=192,
        # cells_per_grid_3d=256
    )

    logger.info(f"lacss_server: loaded model from {modelpath}")

    model.module.detector.min_score = 0.2
    model.module.detector_3d.min_score = 0.2

    logger.debug(f"lacss_server: precompile the model")

    _ = model.predict(np.ones([384, 384, 384, 3]), output_type="_raw")
    _ = model.predict(np.ones([544, 544, 3]), output_type="_raw")

    return model


def show_urls():
    from . import model_urls

    print("Pretrained model files:")
    print("==============================")
    for k, v in model_urls.items():
        print(f"{k}: {v}")
    print()


@app.command()
def main(
    modelpath: Path | None = None,
    port: int = 50051,
    workers: int = 10,
    ip: str = "0.0.0.0",
    local: bool = False,
    token: bool | None = None,
    debug: bool = False,
    compression: bool = True,
    f16: bool = False,
):
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    if modelpath is None:
        show_urls()
        return

    print("server starting ...")

    model = get_predictor(modelpath, f16)

    logger.info(f"lacss_server: default backend is {jax.default_backend()}")

    if jax.default_backend() == "cpu":
        logger.warning(
            f"lacss_server: WARNING: No GPU configuration. This might be very slow ..."
        )

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
        LacssServicer(model),
        server,
    )

    if local:
        server.add_secure_port(f"127.0.0.1:{port}", grpc.local_server_credentials())
    else:
        server.add_insecure_port(f"{ip}:{port}")

    logger.info(f"lacss_server: listening on port {port}")

    print("server starting ... ready")

    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    app()
