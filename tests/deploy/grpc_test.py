import biopb.image as proto
import grpc
import pytest
from biopb.image.utils import serialize_from_numpy

_MAX_MSG_SIZE = 1024 * 1024 * 16


def test_grpc_2d(grpc_channel, test_image_2d):
    pixels = serialize_from_numpy(test_image_2d)

    stub = proto.ObjectDetectionStub(grpc_channel)

    request = proto.DetectionRequest(
        image_data=proto.ImageData(pixels=pixels),
    )

    response = stub.RunDetection(request, timeout=60)

    assert len(response.detections) == 22

    request = proto.DetectionRequest(
        image_data=proto.ImageData(pixels=pixels),
        detection_settings=proto.DetectionSettings(
            nms_iou=0.4,
            scaling_hint=1.0,
        ),
    )

    response = stub.RunDetection(request)

    assert len(response.detections) == 22


def test_grpc_3d(grpc_channel, test_image_3d):
    pixels = serialize_from_numpy(
        test_image_3d[..., None],
        physical_size_x=1.0,
        physical_size_y=1.0,
        physical_size_z=4.0,
    )

    stub = proto.ObjectDetectionStub(grpc_channel)

    request = proto.DetectionRequest(
        image_data=proto.ImageData(pixels=pixels),
        detection_settings=proto.DetectionSettings(
            min_score=0.3,
            nms_iou=0.4,
            scaling_hint=1.0,
        ),
    )

    response = stub.RunDetection(request, timeout=120)

    assert len(response.detections) > 0
