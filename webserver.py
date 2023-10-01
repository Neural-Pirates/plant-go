import base64
import datetime
import logging
import pathlib
import sqlite3
import sys

import cv2
import numpy as np
import orjson
import torch
from aiohttp import ClientSession, web
from torchvision import transforms
from torchvision.models import resnet18

from server.constants import DEPLOYMENT_SERVER_PASS_KEY
from server.ml_utils import load_model, resnet18

cwd = pathlib.Path(__file__).parent.resolve()

logging.basicConfig(
    level=logging.DEBUG,
)

USER_ID = "james"

if not sys.argv[1:]:
    print(f"Usage: {sys.argv[0]} [MINER_ADDRESS]")


MINER_ADDRESS = sys.argv[1]

client_session = None


async def get_client_session():
    # NOTE: This is not thread-safe!

    global client_session

    if client_session is None:
        client_session = ClientSession()

    return client_session


events_database_file = cwd / "events.sqlite"
events_database_connection = sqlite3.connect(events_database_file.as_posix())

events_database_connection.cursor().execute(
    "create table if not exists events (plant text, longitude float, lalitude float)"
)


assets_dir = cwd / "assets"

with open(assets_dir / r"common_output.json", "r") as species_mapping_file:
    species_mapping = orjson.loads(species_mapping_file.read())


model_file = assets_dir / "resnet18_weights_best_acc.tar"
use_gpu = False
model = resnet18(num_classes=1081)

load_model(model, filename=model_file.as_posix(), use_gpu=use_gpu)


routes = web.RouteTableDef()


def has_deployment_authorization(deployment_key=None):
    def inner(f):
        async def wrapper(request):
            return await f(request)
            if deployment_key is None:
                return await f(request)
            else:
                if request.headers.get("Authorization") == deployment_key:
                    return await f(request)
                else:
                    return web.Response(status=401)

        return wrapper

    return inner


@routes.get("/events")
@has_deployment_authorization(f"psk {DEPLOYMENT_SERVER_PASS_KEY}:admin")
async def get_events(request: web.Request):
    plants = events_database_connection.execute("select * from events").fetchall()

    return web.Response(
        body=orjson.dumps(
            list(
                {"plant_id": plant_id, "longitude": longitude, "latitude": latitude}
                for (plant_id, longitude, latitude) in plants
            )
        )
    )


@routes.post("/create_event")
@has_deployment_authorization(f"psk {DEPLOYMENT_SERVER_PASS_KEY}:admin")
async def create_events(request: web.Request):
    data = await request.json()

    if not data or any(
        field not in data for field in ["plant_id", "longitude", "latitude"]
    ):
        raise web.HTTPBadRequest(reason="Invalid data provided.")

    events_database_connection.execute(
        "insert into events values (?, ?, ?)",
        (data["plant_id"], data["longitude"], data["latitude"]),
    )
    events_database_connection.commit()
    return web.Response()


@routes.get("/fetch_chain")
@has_deployment_authorization(f"psk {DEPLOYMENT_SERVER_PASS_KEY}")
async def fetch_chain(request: web.Request):
    session = await get_client_session()

    async with session.get(f"http://{MINER_ADDRESS}/chain") as resp:
        return web.Response(body=await resp.read(), status=resp.status)


@routes.post("/register_prediction")
@has_deployment_authorization(f"psk {DEPLOYMENT_SERVER_PASS_KEY}")
async def register_prediction(request: web.Request):
    response_data = await request.json()

    if not response_data or any(field not in response_data for field in ["caught_at"]):
        caught_at = response_data.get("caught_at")

        if not caught_at or any(field not in caught_at for field in ["lat", "lon"]):
            raise web.HTTPBadRequest(
                reason="Invalid data provided: cannot decide where the plant was caught."
            )

        raise web.HTTPBadRequest(reason="Invalid data provided: cannot identify plant.")

    caught_on = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    data = base64.b64decode(response_data.get("image"))

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
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

    args = {
        "plant_id": predicted_class,
        "client_id": USER_ID,
        "caught_at": response_data["caught_at"],
        "caught_on": caught_on,
    }

    session = await get_client_session()

    async with session.post(
        f"http://{MINER_ADDRESS}/transactions/new", json=args
    ) as resp:
        if resp.status != 201:
            raise web.HTTPBadRequest(reason="Could not register transaction.")

    async with session.get(f"http://{MINER_ADDRESS}/mine") as _:
        ...

    return web.Response(
        body=orjson.dumps(
            {
                "plant": {"id": predicted_class, "name": predicted_species},
                "client_id": USER_ID,
                "caught_at": response_data["caught_at"],
                "caught_on": caught_on,
            }
        )
    )


def main():
    app = web.Application(client_max_size=1024**2 * 100)
    app.add_routes(routes)

    return app


if __name__ == "__main__":
    web.run_app(main(), host="0.0.0.0", port=8080)
