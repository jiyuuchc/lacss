from pathlib import Path

import pytest
import grpc

MODULE_DIR = Path(__file__).parent

@pytest.fixture(scope='package')
def model_file():
    import urllib.request

    cache_dir = Path.home() / ".cache" / "lacss"
    model_file = cache_dir / "lacss3_small"

    if not model_file.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        urllib.request.urlretrieve(
            "https://huggingface.co/jiyuuchc/lacss3-small/resolve/main/lacss3-small?download=true",
            model_file,
        )

    yield model_file


@pytest.fixture(scope='package')
def model(model_file):
    from lacss.deploy.predict import Predictor

    predictor = Predictor(model_file)

    yield predictor

    del predictor


@pytest.fixture(scope='package')
def model_f16(model_file):
    from lacss.deploy.predict import Predictor

    predictor = Predictor(model_file, f16=True)

    yield predictor

    del predictor


@pytest.fixture(scope='package')
def test_image_2d():
    import tifffile
    img = tifffile.imread(MODULE_DIR / "test_data" / "test_2d.tif")

    yield img

    del img


@pytest.fixture(scope='package')
def test_image_3d():
    import tifffile
    img = tifffile.imread(MODULE_DIR / "test_data" / "test_3d.tif")

    yield img

    del img


@pytest.fixture(scope='package')
def grpc_server(model_f16):
    from concurrent import futures
    import lacss.deploy.proto as proto

    from lacss.deploy.remote_server import LacssServicer

    _MAX_MSG_SIZE = 1024 * 1024 * 16
    endpoint = "127.0.0.1:50051"

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=(("grpc.max_receive_message_length", _MAX_MSG_SIZE),),
    )
    proto.add_LacssServicer_to_server(
        LacssServicer(model_f16), server
    )
    server.add_insecure_port(endpoint)

    server.start()

    yield endpoint

    server.stop(grace=None)


@pytest.fixture()
def grpc_channel(grpc_server):
    with grpc.insecure_channel(grpc_server) as channel:
        yield channel
