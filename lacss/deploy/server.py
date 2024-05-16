import struct
import sys
from pathlib import Path
from zipfile import ZipFile

import jax
import jax.numpy as jnp
import numpy as np
import typer

import lacss.deploy.proto.lacss_pb2 as lacss_pb2
from lacss.ops import patches_to_label, sorted_non_max_suppression

from .predict import Predictor

app = typer.Typer(pretty_exceptions_enable=False)

def read_input(st = sys.stdin.buffer):
    msg_size = st.read(4)
    msg_size = struct.unpack(">i", msg_size)[0]

    msg = lacss_pb2.Input()
    msg.ParseFromString(st.read(msg_size))

    image = msg.image
    np_img = np.frombuffer(image.data, dtype=">f4").astype("float32")
    np_img = np_img.reshape(image.height, image.width, image.channel)
    # np_img = np_img.transpose(2, 1, 0)

    # import imageio.v2 as imageio
    # imageio.imwrite("tmpin.tif", np_img)

    return np_img, msg.settings

def img_to_msg(msg, img):
    img = np.ascontiguousarray(img, dtype=">i2") # match java format

    msg.height = img.shape[0]
    msg.width = img.shape[1]
    msg.data = img.tobytes()

    assert len(msg.data) == msg.height * msg.width * 2

def write_result(label, score, st = sys.stdout.buffer):

    msg = lacss_pb2.Result()
    img_to_msg(msg.score, (score * 1000).astype(np.int16))
    img_to_msg(msg.label, label)

    msg_size_bits = struct.pack(">i", msg.ByteSize())

    st.write(msg_size_bits)
    st.write(msg.SerializeToString())

def write_polygon_result(polygons, scores, st = sys.stdout.buffer):
    msg = lacss_pb2.PolygonResult()
    for polygon, score in zip(polygons, scores):
        if len(polygon) > 1:
            polygon_msg = LacssMsg.Polygon()
            polygon_msg.score = score
            for p in polygon:
                point = LacssMsg.Point()
                point.x = p[0]
                point.y = p[1]
                polygon_msg.points.append(point)
            msg.polygons.append(polygon_msg)
    
    msg_size_bits = struct.pack(">i", msg.ByteSize())

    st.write(msg_size_bits)
    st.write(msg.SerializeToString())

@app.command()
def main(modelpath: Path):
    modelpath = str(modelpath)
    if "!" in modelpath: # in a jar
        jarfile, rscpath = modelpath.split("!")
        with ZipFile(jarfile) as jar:
            with jar.open(rscpath) as modelfile:
                import pickle
                modelpath = pickle.load(modelfile)

    model = Predictor(modelpath)
    model.detector.test_max_output = 512
    model.detector.test_min_score = 0

    print(f"lacss_server: loaded model from {modelpath}", file=sys.stderr)
    print(f"lacss_server: default backend is {jax.default_backend()}", file=sys.stderr)
    if (jax.default_backend() == "cpu"):
        print(f"lacss_server: WARNING: No GPU configuration. This might be very slow ...", file=sys.stderr)

    # cnt = 0

    while True:
        img, settings = read_input()
        # img = img - img.min()
        # img = img / img.max()
        img -= img.mean()
        img = img / img.std()

        print(f"received image {img.shape}", file=sys.stderr)

        preds = model.predict(
            img, 
            min_area=settings.min_cell_area,
            remove_out_of_bound=settings.remove_out_of_bound,
            scaling=settings.scaling,
            nms_iou=settings.nms_iou,
            score_threshold=settings.detection_threshold,
            segmentation_threshold=settings.segmentation_threshold,
            output_type="contour" if settings.return_polygon else "label"
        )

        if settings.return_polygon:

            write_polygon_result(
                preds["pred_contours"],
                preds["pred_scores"],
            )

        else:

            label = preds["pred_label"].astype("int")
            scores = preds["pred_scores"]

            assert label.max() == len(scores)

            assert len(label.shape) == 2

            # score image
            score_img = scores[label - 1]
            score_img *= (label > 0)

            assert score_img.shape == label.shape

            write_result(label, score_img)

        # imageio.imwrite(f"p_{cnt}.tif", np.asarray(label))
        # cnt+=1

if __name__ == "__main__":
    app()
