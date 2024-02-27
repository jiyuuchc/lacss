import logging
import struct
import sys
from concurrent import futures
from pathlib import Path
from zipfile import ZipFile

import grpc
import jax
import jax.numpy as jnp
import numpy as np
import typer

from .predict import Predictor
from .proto import lacss_pb2, lacss_pb2_grpc

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

@app.command()
def main(
    modelpath: Path,
    port: int = 50051,
    workers: int = 10,
    debug: bool = False,
):
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
    # logging.basicConfig(level=logging.INFO)

    model = Predictor(modelpath)
    model.detector.test_max_output = 512
    model.detector.test_min_score = 0

    logging.info(f"lacss_server: loaded model from {modelpath}")
    logging.info(f"lacss_server: default backend is {jax.default_backend()}")
    if (jax.default_backend() == "cpu"):
        logging.warn(f"lacss_server: WARNING: No GPU configuration. This might be very slow ...")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers))
    lacss_pb2_grpc.add_LacssServicer_to_server(LacssServicer(model), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    app()
