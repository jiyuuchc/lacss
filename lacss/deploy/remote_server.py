import logging
from concurrent import futures
from pathlib import Path

import grpc
import jax
import numpy as np
import typer

from .predict import Predictor
from .proto import lacss_pb2, lacss_pb2_grpc

_AUTH_HEADER_KEY = "authorization"

app = typer.Typer(pretty_exceptions_enable=False)


def process_input(msg):
    image = msg.image
    np_img = np.frombuffer(image.data, dtype=">f4").astype("float32")
    np_img = np_img.reshape(image.height, image.width, image.channel)

    return np_img, msg.settings


def process_result(polygons, scores):
    msg = lacss_pb2.PolygonResult()
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
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid signature")

        self._abort_handler = grpc.unary_unary_rpc_method_handler(abort)
        self.token = token

    def intercept_service(self, continuation, handler_call_details):
        # Example HandlerCallDetails object:
        #     _HandlerCallDetails(
        #       method=...,
        #       invocation_metadata=...)
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

        img -= img.mean()
        img = img / img.std()

        logging.debug(f"received image {img.shape}")

        logging.debug(f"making prediction")

        preds = self.model.predict(
            img,
            min_area=settings.min_cell_area,
            remove_out_of_bound=settings.remove_out_of_bound,
            scaling=settings.scaling,
            nms_iou=settings.nms_iou,
            score_threshold=settings.detection_threshold,
            segmentation_threshold=settings.segmentation_threshold,
            output_type="contour",
        )

        logging.debug(f"formatting reply")

        result = process_result(
            preds["pred_contours"],
            preds["pred_scores"],
        )

        logging.debug(f"Reply with message of size {result.ByteSize()}")

        return result


def get_predictor(modelpath):
    model = Predictor(modelpath)

    logging.info(f"lacss_server: loaded model from {modelpath}")

    model.detector.test_max_output = 512  # FIXME good default?
    model.detector.test_min_score = 0.4

    return model


@app.command()
def main(
    modelpath: Path,
    port: int = 50051,
    workers: int = 10,
    ip: str = "0.0.0.0",
    local: bool = False,
    token: bool = False,
    debug: bool = False,
):
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)

    model = get_predictor(modelpath)

    logging.info(f"lacss_server: default backend is {jax.default_backend()}")

    if jax.default_backend() == "cpu":
        logging.warn(
            f"lacss_server: WARNING: No GPU configuration. This might be very slow ..."
        )

    if token:
        import secrets

        token = secrets.token_urlsafe()

        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=workers),
            interceptors=(TokenValidationInterceptor(token),),
        )

        print("===================================")
        print(f"For access use token: {token}")
        print("===================================")
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
