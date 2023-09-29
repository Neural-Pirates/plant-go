import enum
import uuid

import orjson
from aiohttp import ClientSession, web

from blockchain.utils import Blockchain

identifier = uuid.uuid4().hex

routes = web.RouteTableDef()
blockchain = Blockchain()

client_session = None


async def get_client_session():
    # NOTE: This is not thread-safe!

    global client_session

    if client_session is None:
        client_session = ClientSession()

    return client_session


class BlockchainStates(enum.Enum):
    NEW_BLOCK_FORMED = "new_block_formed"
    CHAIN_REPLACED = "chain_replaced"
    CHAIN_AUTHORITATIVE = "chain_authoritative"
    NODE_UPDATED = "node_updated"


@routes.get("/mine")
async def mine(request: web.Request):
    last_block = blockchain.last_block
    proof = await blockchain.async_proof_of_work(last_block)

    blockchain.new_transaction(
        sender="0",
        recipient=identifier,
        amount=1,
    )

    previous_hash = blockchain.hash(last_block)
    block = blockchain.new_block(proof, previous_hash)

    response = {
        "message": BlockchainStates.NEW_BLOCK_FORMED.name,
        "index": block["index"],
        "transactions": block["transactions"],
        "proof": block["proof"],
        "previous_hash": block["previous_hash"],
    }
    return web.Response(body=orjson.dumps(response).decode(), status=200)


@routes.post("/transactions/new")
async def new_transaction(request: web.Request):
    values = await request.json()

    if not all(k in values for k in Blockchain.transaction_fields):
        raise web.HTTPBadRequest(
            "One or more of the required field are missing for a chain."
        )

    index = blockchain.new_transaction(**values)

    response = {"message": f"Transaction will be added to Block {index}"}
    return web.Response(body=orjson.dumps(response).decode(), status=201)


@routes.get("/chain")
async def full_chain(request: web.Request):
    response = {
        "chain": blockchain.chain,
        "length": len(blockchain.chain),
    }
    return web.Response(body=orjson.dumps(response).decode(), status=200)


@routes.post("/nodes/register")
async def register_nodes(request: web.Request):
    values = await request.json()

    nodes = values.get("nodes")
    if nodes is None:
        raise web.HTTPBadRequest()

    for node in nodes:
        blockchain.register_node(node)

    response = {
        "message": BlockchainStates.NODE_UPDATED.value,
        "total_nodes": list(blockchain.nodes),
    }
    return web.Response(body=orjson.dumps(response).decode(), status=201)


@routes.get("/nodes/resolve")
async def consensus(request: web.Request):
    session = await get_client_session()

    replaced = await blockchain.resolve_conflicts(session)

    if replaced:
        response = {
            "message": BlockchainStates.CHAIN_REPLACED.value,
            "new_chain": blockchain.chain,
        }
    else:
        response = {
            "message": BlockchainStates.CHAIN_AUTHORITATIVE.value,
            "chain": blockchain.chain,
        }

    return web.Response(body=orjson.dumps(response).decode(), status=200)