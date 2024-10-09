import logging
from concurrent import futures
from pathlib import Path

import grpc
import jax
import numpy as np
import typer

from .proto import lacss_pb2, lacss_pb2_grpc

_AUTH_HEADER_KEY = "authorization"

app = typer.Typer(pretty_exceptions_enable=False)

_MAX_MSG_SIZE=1024*1024*64
_TARGET_CELL_SIZE=32

def process_input(msg):
    image = msg.image
    settings = msg.settings

    if image.voxel_dim_x != 0 and image.voxel_dim_y != 0 and settings.cell_size_hint != 0:
        scale_x = _TARGET_CELL_SIZE / settings.cell_size_hint * image.voxel_dim_x
        scale_y = _TARGET_CELL_SIZE / settings.cell_size_hint * image.voxel_dim_y
        if msg.voxel_dim_z != 0:
            scale_z = _TARGET_CELL_SIZE / settings.cell_size_hint / 3 * image.voxel_dim_z
        else:
            scale_z = scale_x
    elif settings.scaling != 0:
        scale_x = scale_y = scale_z = settings.scaling
    else:
        scale_x = scale_y = scale_z = 1

    logging.debug(f"Requested rescaling factor is {(scale_z, scale_y, scale_x)}")

    np_img = np.frombuffer(image.data, dtype=">f4").astype("float32")
    if image.depth == 0 or image.depth == 1:
        np_img = np_img.reshape(image.height, image.width, image.channel)
        shape_hint = (
            int(scale_y * image.height + .5),
            int(scale_x * image.width + .5),
        )
    else:
        np_img = np_img.reshape(image.depth, image.height, image.width, image.channel)
        shape_hint = (
            int(scale_z * image.depth +.5),
            int(scale_y * image.height + .5),
            int(scale_x * image.width + .5),
        )

    return np_img, shape_hint


def process_result(preds, is3d):
    msg = lacss_pb2.PolygonResult()

    scores = preds["pred_scores"]
    if is3d:
        bboxes = preds["pred_bboxes"]
        centroids = bboxes.reshape(-1, 2, 3).mean(1)

        for loc, score in zip(centroids, preds["pred_scores"]):
            polygon_msg = lacss_pb2.Polygon()
            polygon_msg.score = score
            point = lacss_pb2.Point()
            point.z, point.y, point.x = loc
            polygon_msg.points.append(point)

            msg.polygons.append(polygon_msg)

        # from skimage.measure import regionprops
        # label = preds["pred_label"].astype(int)

        # for rp, score in zip(regionprops(label), scores):
        #     polygon_msg = lacss_pb2.Polygon()
        #     polygon_msg.score = score
        #     point = lacss_pb2.Point()
        #     point.z, point.y, point.x = rp.centroid
        #     polygon_msg.points.append(point)
    
        #     msg.polygons.append(polygon_msg)

    else:
        polygons = preds["pred_contours"]
        for polygon, score in zip(polygons, scores):
            if len(polygon) > 0:
                polygon_msg = lacss_pb2.Polygon()
                polygon_msg.score = score
                for p in polygon:
                    point = lacss_pb2.Point()
                    point.x = p[0]
                    point.y = p[1]
                    polygon_msg.points.append(point)
                msg.polygons.append(polygon_msg)

    return msg


class TokenValidationInterceptor(grpc.ServerInterceptor):
    def __init__(self, token):
        def abort(ignored_request, context):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid token signature")

        self._abort_handler = grpc.unary_unary_rpc_method_handler(abort)
        self.token = token

    def intercept_service(self, continuation, handler_call_details):
        expected_metadata = (_AUTH_HEADER_KEY, f"Bearer {self.token}")
        if self.token is None or expected_metadata in handler_call_details.invocation_metadata:
            return continuation(handler_call_details)
        else:
            return self._abort_handler


class LacssServicer(lacss_pb2_grpc.LacssServicer):
    def __init__(self, model):
        self.model = model

    def RunDetection(self, request, context):
        img, shape_hint = process_input(request)
        settings = request.settings
        is3d = img.ndim == 4

        logging.debug(f"received image {img.shape}")

        # dont' reshape image if the request is almost same as the orginal 
        rel_diff = np.abs(np.array(shape_hint) / img.shape[:-1] - 1)
        if (rel_diff < 0.1).all():
            shape_hint = None 

        preds = self.model.predict(
            img,
            reshape_to = shape_hint,
            min_area=settings.min_cell_area,
            nms_iou=settings.nms_iou,
            score_threshold=settings.detection_threshold,
            segmentation_threshold=settings.segmentation_threshold,
            output_type="bbox" if is3d else "contour" ,
        )

        result = process_result(preds, is3d)

        logging.debug(f"Reply with message of size {result.ByteSize()}")

        return result


def get_predictor(modelpath):
    from .predict import Predictor

    model = Predictor(modelpath)

    logging.info(f"lacss_server: loaded model from {modelpath}")

    model.module.detector.max_output = 512  # FIXME good default?
    model.module.detector.min_score = 0.2

    logging.debug(f"lacss_server: precompile for most common shapes")
    _ = model.predict(np.ones([544,544,3]))
    _ = model.predict(np.ones([64,256,256,3]))

    return model


def show_urls():
    from .predict import model_urls
    
    print("Pretrained model files:")
    print("==============================")
    for k, v in model_urls.items():
        print(f"{k}: {v}")
    print()


@app.command()
def main(
    modelpath: Path|None = None,
    port: int = 50051,
    workers: int = 10,
    ip: str = "0.0.0.0",
    local: bool = False,
    token: bool|None = None,
    debug: bool = False,
    compression: bool = True,
):
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)

    if modelpath is None:
        show_urls()
        return

    model = get_predictor(modelpath)

    logging.info(f"lacss_server: default backend is {jax.default_backend()}")

    if jax.default_backend() == "cpu":
        logging.warn(
            f"lacss_server: WARNING: No GPU configuration. This might be very slow ..."
        )

    if token is None:
        token = not local
    if token:
        import secrets

        token = secrets.token_urlsafe(32)

        print()
        print("COPY THE TOKEN BELOW FOR ACCESS.")
        print("=======================================================================")
        print(f"{token}")
        print("=======================================================================")
        print()
    else:
        token = None

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=workers),
        compression=grpc.Compression.Gzip if compression else grpc.Compression.NoCompression,
        interceptors=(TokenValidationInterceptor(token),),
        options=(("grpc.max_receive_message_length", _MAX_MSG_SIZE),),
    )

    lacss_pb2_grpc.add_LacssServicer_to_server(LacssServicer(model), server)

    if local:
        server.add_secure_port(f"localhost:{port}", grpc.local_server_credentials())
    else:
        server.add_insecure_port(f"{ip}:{port}")

    logging.info(f"lacss_server: listening on port {port}")

    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    app()