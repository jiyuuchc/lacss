import importlib.machinery
import logging
import threading
import traceback
import types
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np
import torch
import typer
from predict import *
from predict import _normalize

from . import proto
from .common import TokenValidationInterceptor, decode_image

app = typer.Typer(pretty_exceptions_enable=False)

_MAX_MSG_SIZE = 1024 * 1024 * 128


def process_input(request: proto.DetectionRequest):
    pixels = request.image_data.pixels
    settings = request.detection_settings

    image = decode_image(pixels).squeeze(0)

    return image


def process_result(masks, image):
    import cv2
    from skimage.measure import regionprops

    response = proto.DetectionResponse()

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


class SegformerServicer(proto.LacssServicer):
    def __init__(self, gpu, model_path, model_path2):
        self.model_path = model_path
        self.model_path2 = model_path2
        self.device = "cuda:0" if gpu else "cpu"
        self._lock = threading.RLock()

    def predict(self, img_data):
        hflip_tta = HorizontalFlip()
        vflip_tta = VerticalFlip()

        if img_data.shape[-1] == 1:
            img_data = np.repeat(img_data, 3, axis=-1)
        elif img_data.shape[-1] == 2:
            raise ValueError("Model accept either 1-channel or 3-channel images.")

        H, W = img_data.shape[:2]
        H1 = (H - 1) // 32 * 32 + 32
        W1 = (W - 1) // 32 * 32 + 32
        img_data = np.pad(img_data, [[0, H1 - H], [0, W1 - W], [0, 0]])

        img_data = _normalize(img_data)
        img_data = img_data.astype("float32")
        img_data = np.moveaxis(img_data, -1, 0)
        img_data = img_data[None, ...]

        model = torch.load(self.model_path, map_location=self.device)
        model.eval()

        img_data = torch.from_numpy(img_data).to(self.device)
        img_size = img_data.shape[-1] * img_data.shape[-2]

        if img_size < 1150000 and 900000 < img_size:
            overlap = 0.5
        else:
            overlap = 0.6

        with torch.no_grad():
            img0 = img_data
            outputs0 = sliding_window_inference(
                img0,
                512,
                4,
                model,
                padding_mode="reflect",
                mode="gaussian",
                overlap=overlap,
                device="cpu",
            )
            outputs0 = outputs0.cpu().squeeze()

            if img_size < 2000 * 2000 or img_size > 5000 * 5000:

                model.load_state_dict(
                    torch.load(self.model_path2, map_location=self.device)
                )
                model.eval()

                img2 = hflip_tta.apply_aug_image(img_data, apply=True)
                outputs2 = sliding_window_inference(
                    img2,
                    512,
                    4,
                    model,
                    padding_mode="reflect",
                    mode="gaussian",
                    overlap=overlap,
                    device="cpu",
                )
                outputs2 = hflip_tta.apply_deaug_mask(outputs2, apply=True)
                outputs2 = outputs2.cpu().squeeze()

                outputs = torch.zeros_like(outputs0)
                outputs[0] = (outputs0[0] + outputs2[0]) / 2
                outputs[1] = (outputs0[1] - outputs2[1]) / 2
                outputs[2] = (outputs0[2] + outputs2[2]) / 2

            else:
                # Hflip TTA
                img2 = hflip_tta.apply_aug_image(img_data, apply=True)
                outputs2 = sliding_window_inference(
                    img2,
                    512,
                    4,
                    model,
                    padding_mode="reflect",
                    mode="gaussian",
                    overlap=overlap,
                    device="cpu",
                )
                outputs2 = hflip_tta.apply_deaug_mask(outputs2, apply=True)
                outputs2 = outputs2.cpu().squeeze()
                img2 = img2.cpu()

                ##################
                #                #
                #    ensemble    #
                #                #
                ##################

                model.load_state_dict(
                    torch.load(args.model_path2, map_location=args.device)
                )
                model.eval()

                img1 = img_data
                outputs1 = sliding_window_inference(
                    img1,
                    512,
                    4,
                    model,
                    padding_mode="reflect",
                    mode="gaussian",
                    overlap=overlap,
                    device="cpu",
                )
                outputs1 = outputs1.cpu().squeeze()

                # Vflip TTA
                img3 = vflip_tta.apply_aug_image(img_data, apply=True)
                outputs3 = sliding_window_inference(
                    img3,
                    512,
                    4,
                    model,
                    padding_mode="reflect",
                    mode="gaussian",
                    overlap=overlap,
                    device="cpu",
                )
                outputs3 = vflip_tta.apply_deaug_mask(outputs3, apply=True)
                outputs3 = outputs3.cpu().squeeze()
                img3 = img3.cpu()

                # Merge Results
                outputs = torch.zeros_like(outputs0)
                outputs[0] = (outputs0[0] + outputs1[0] + outputs2[0] - outputs3[0]) / 4
                outputs[1] = (outputs0[1] + outputs1[1] - outputs2[1] + outputs3[1]) / 4
                outputs[2] = (outputs0[2] + outputs1[2] + outputs2[2] + outputs3[2]) / 4

            pred_mask = post_process(outputs.squeeze(0).cpu().numpy(), self.device)

            return pred_mask

    def RunDetection(self, request, context):
        with self._lock:

            logging.info(f"Received message of size {request.ByteSize()}")

            try:
                image = process_input(request)

                if image.ndim == 4:
                    raise ValueError(
                        f"Model does not support 3D input, got shape {image.shape}"
                    )

                logging.info(f"received image {image.shape}")

                preds = self.predict(image)

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
    port: int = 50051,
    workers: int = 10,
    ip: str = "0.0.0.0",
    local: bool = False,
    token: bool | None = None,
    debug: bool = False,
    compression: bool = True,
    gpu: bool = True,
    model_path: Path = "./main_model.pt",
    model_path2: Path = "./sub_model.pth",
):
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)

    print("server starting ...")

    # loader = importlib.machinery.SourceFileLoader('predict', str(predict_py_file))
    # mod = types.ModuleType( loader.name )
    # loader.exec_module(mod)

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
        SegformerServicer(gpu, model_path, model_path2),
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
