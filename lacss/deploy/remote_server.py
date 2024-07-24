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


def process_input(msg):
    image = msg.image
    np_img = np.frombuffer(image.data, dtype=">f4").astype("float32")
    if image.depth == 0 or image.depth == 1:
        np_img = np_img.reshape(image.height, image.width, image.channel)
    else:
        np_img = np_img.reshape(image.depth, image.height, image.width, image.channel)

    return np_img, msg.settings


def process_result(preds, is3d):
    msg = lacss_pb2.PolygonResult()

    scores = preds["pred_scores"]
    if is3d:
        from skimage.measure import regionprops
        label = preds["pred_label"].astype(int)

        for rp, score in zip(regionprops(label), scores):
            polygon_msg = lacss_pb2.Polygon()
            polygon_msg.score = score
            point = lacss_pb2.Point()
            point.z, point.y, point.x = rp.centroid
            polygon_msg.points.append(point)
    
            msg.polygons.append(polygon_msg)

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
        if expected_metadata in handler_call_details.invocation_metadata:
            return continuation(handler_call_details)
        else:
            return self._abort_handler


class LacssServicer(lacss_pb2_grpc.LacssServicer):
    def __init__(self, model):
        self.model = model

    def RunDetection(self, request, context):
        img, settings = process_input(request)
        is3d = img.ndim == 4
        logging.debug(f"received image {img.shape}")

        preds = self.model.predict(
            img,
            min_area=settings.min_cell_area,
            scaling=settings.scaling,
            nms_iou=settings.nms_iou,
            score_threshold=settings.detection_threshold,
            segmentation_threshold=settings.segmentation_threshold,
            output_type="label" if is3d else "contour" ,
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

        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=workers),
            interceptors=(TokenValidationInterceptor(token),),
        )

        print()
        print("COPY THE TOKEN BELOW FOR ACCESS.")
        print("=======================================================================")
        print(f"{token}")
        print("=======================================================================")
        print()
    else:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers))

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
