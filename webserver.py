import pathlib
import random
import sqlite3
import string
import sys
import tempfile
import time
import uuid

import cv2
import orjson
import torch
from aiohttp import web
from torchvision import transforms
from torchvision.models import resnet18

from server.constants import DEPLOYMENT_SERVER_PASS_KEY
from server.ml_utils import load_model, resnet18

cwd = pathlib.Path(__file__).parent.resolve()


if not sys.argv[1:]:
    print(f"Usage: {sys.argv[0]} [MINERS]")


MINER_ADDRESS = sys.argv[1:]


prediction_database_file = cwd / "pending_predictions.sqlite"
prediction_database_connection = sqlite3.connect(prediction_database_file.as_posix())

prediction_database_connection.cursor().execute(
    "create table if not exists prediction (plant integer, valid_upto integer, token text)"
)


events_database_file = cwd / "events.sqlite"
events_database_connection = sqlite3.connect(events_database_file.as_posix())

events_database_connection.cursor().execute(
    "create table if not exists events (plant integer, longitude float, lalitude float)"
)


assets_dir = cwd / "assets"

with open(assets_dir / r"common_output.json", "r") as species_mapping_file:
    species_mapping = orjson.loads(species_mapping_file.read())


model_file = assets_dir / "resnet18_weights_best_acc.tar"
use_gpu = False
model = resnet18(num_classes=1081)

load_model(model, filename=model_file.as_posix(), use_gpu=use_gpu)


routes = web.RouteTableDef()
predictions = set()


class Prediction:
    MAX_TIME_VALID = 3600

    def __init__(self, plant_id: str):
        self.plant_id = plant_id
        self.valid_upto = time.time() + Prediction.MAX_TIME_VALID

        self.prediction_token = uuid.uuid4().hex


def create_prediction(plant_id, *, until=3600):
    prediction_token = uuid.uuid4().hex
    valid_upto = int(time.time() + until) * 1000

    prediction_database_connection.execute(
        "insert into prediction values (?, ?, ?)",
        (plant_id, prediction_token, valid_upto),
    )
    prediction_database_connection.commit()
    return prediction_token


def get_prediction(prediction_token: str):
    return prediction_database_connection.execute(
        "select * from prediction where token match ? and valid_upto > ?",
        (prediction_token, int(time.time()) * 1000),
    ).fetchone()


def has_deployment_authorization(deployment_key=None):
    def inner(f):
        async def wrapper(request):
            if deployment_key is None:
                return await f(request)
            else:
                if request.headers.get("Authorization") == deployment_key:
                    return await f(request)
                else:
                    return web.Response(status=401)

        return wrapper

    return inner


@routes.get("/api")
@has_deployment_authorization(f"psk {DEPLOYMENT_SERVER_PASS_KEY}")
async def api(request: web.Request):
    return web.Response(text="Hello, world")


@routes.post("/predict_from_image")
@has_deployment_authorization(f"psk {DEPLOYMENT_SERVER_PASS_KEY}")
async def predict_from_image(request: web.Request):
    data = await request.read()

    if not data:
        raise web.HTTPBadRequest(reason="No data provided.")

    tf = pathlib.Path(tempfile.gettempdir()) / (
        "".join(random.choices(string.ascii_uppercase + string.digits, k=10)) + ".png"
    )

    with open(tf.as_posix(), "wb") as f:
        f.write(data)

    img = cv2.imread(tf.as_posix())

    if img is None:
        raise web.HTTPBadRequest(reason="Invalid image provided.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
        img
    )
    img = img.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        output = model(img)

    predicted_class = torch.argmax(output, dim=1).item()
    predicted_species = species_mapping.get(str(predicted_class), "unknown")

    tf.unlink()

    prediction_token = create_prediction(predicted_class)

    return web.Response(
        body=orjson.dumps(
            {
                "identified_species": predicted_species,
                "cls_index": predicted_class,
                "token": prediction_token,
            }
        )
    )


@routes.post("/register_prediction")
@has_deployment_authorization(f"psk {DEPLOYMENT_SERVER_PASS_KEY}")
async def register_prediction(request: web.Request):
    token = request.query.get("token")

    if token is None:
        raise web.HTTPBadRequest(reason="No token was passed.")

    data = get_prediction(token)

    if data is None:
        raise web.HTTPBadRequest(
            reason="Either the prediciton does not exist or the prediction has expired."
        )

    ...


app = web.Application(client_max_size=1024**2 * 100)
app.add_routes(routes)


web.run_app(app, host="0.0.0.0", port=8080)
