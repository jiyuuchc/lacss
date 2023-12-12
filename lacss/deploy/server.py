import struct
import sys
from pathlib import Path
from zipfile import ZipFile

import jax
import jax.numpy as jnp
import numpy as np
import typer

import lacss.deploy.lacss_pb2 as LacssMsg
from lacss.ops import patches_to_label, sorted_non_max_suppression

from .predict import Predictor

app = typer.Typer(pretty_exceptions_enable=False)

def read_input(st = sys.stdin.buffer):
    msg_size = st.read(4)
    msg_size = struct.unpack(">i", msg_size)[0]

    msg = LacssMsg.Input()
    msg.ParseFromString(st.read(msg_size))

    image = msg.image
    np_img = np.frombuffer(image.data, dtype=">f4").astype("float32")
    np_img = np_img.reshape(image.height, image.width, image.channel)
    # np_img = np_img.transpose(2, 1, 0)

    import imageio.v2 as imageio
    imageio.imwrite("tmpin.tif", np_img)

    return np_img, msg.settings

def img_to_msg(msg, img):
    img = np.ascontiguousarray(img, dtype=">i2") # match java format

    msg.height = img.shape[0]
    msg.width = img.shape[1]
    msg.data = img.tobytes()

    assert len(msg.data) == msg.height * msg.width * 2

def write_result(label, score, st = sys.stdout.buffer):
    if len(label.shape) != 2 :
        raise ValueError(f"Expect 2D array as label. Got array of {label.shape}")

    msg = LacssMsg.Result()
    img_to_msg(msg.score, (score * 1000).astype(np.int16))
    img_to_msg(msg.label, label)

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
    model.detector.test_min_score = 0.2

    print(f"lacss_server: loaded model from {modelpath}", file=sys.stderr)
    print(f"lacss_server: default backend is {jax.default_backend()}", file=sys.stderr)
    if (jax.default_backend() != "cpu"):
        print(f"lacss_server: WARNING: No GPU configuration. This might be very slow ...", file=sys.stderr)

    # cnt = 0

    while True:
        img, settings = read_input()
        img = img - img.min()
        # img = img / img.max()
        img = img / img.std()

        print(f"received image {img.shape}", file=sys.stderr)

        preds = model.predict(
            img, 
            min_area=settings.min_cell_area,
            remove_out_of_bound=settings.remove_out_of_bound,
            scaling=settings.scaling,
        )

        # boxed based nms
        # _, _, mask = sorted_non_max_suppression(
        #     preds["pred_scores"],
        #     preds["pred_bboxes"],
        #     -1, # max number of output
        #     threshold=settings.nms_iou,
        #     min_score=settings.detection_threshold,
        #     return_selection=True,
        # )

        label = np.asarray(patches_to_label(
            preds, 
            img.shape[:2],
            score_threshold=settings.detection_threshold,
            threshold=settings.segmentation_threshold,
        ).astype(int))

        # score image
        score = np.asarray(preds["pred_scores"])[label]
        assert score.shape == label.shape

        # make label continous
        k = np.unique(label)
        v = np.arange(len(k))
        mapping_ar = np.zeros(k.max() + 1, dtype=np.uint16)
        mapping_ar[k] = v
        label = mapping_ar[label]

        write_result(label, score)

        # imageio.imwrite(f"p_{cnt}.tif", np.asarray(label))
        # cnt+=1

if __name__ == "__main__":
    app()
