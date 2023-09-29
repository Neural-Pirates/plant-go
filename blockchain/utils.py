import asyncio
import hashlib
import json
from time import time
from urllib.parse import urlparse

import aiohttp


def valid_hash_proof(guess_hash):
    return guess_hash[:4] == "0000"


def valid_proof(last_proof, proof, last_hash):
    guess = f"{last_proof}{proof}{last_hash}".encode()
    return valid_hash_proof(hashlib.sha256(guess).hexdigest())


class Blockchain:
    transaction_fields = ["client_id", "plant_id"]

    def __init__(self):
        self.current_transactions = []
        self.chain = []
        self.nodes = set()

        self.new_block(previous_hash="1", proof=100)

    def register_node(self, address):
        parsed_url = urlparse(address)
        if parsed_url.netloc:
            self.nodes.add(parsed_url.netloc)
        elif parsed_url.path:
            self.nodes.add(parsed_url.path)
        else:
            raise ValueError("Invalid URL")

    def valid_chain(self, chain):
        last_block = chain[0]
        current_index = 1

        while current_index < len(chain):
            block = chain[current_index]
            last_block_hash = self.hash(last_block)
            if block["previous_hash"] != last_block_hash:
                return False

            if not valid_proof(last_block["proof"], block["proof"], last_block_hash):
                return False

            last_block = block
            current_index += 1

        return True

    async def resolve_conflicts(self, session: aiohttp.ClientSession):
        neighbours = self.nodes
        new_chain = None

        max_length = len(self.chain)

        for node in neighbours:
            async with session.get(f"http://{node}/chain") as response:
                response_json = await response.json()

                if response.status == 200:
                    length = response_json["length"]
                    chain = response_json["chain"]

                    if length > max_length and self.valid_chain(chain):
                        max_length = length
                        new_chain = chain

        if new_chain:
            self.chain = new_chain
            return True

        return False

    def new_block(self, proof, previous_hash):
        block = {
            "index": len(self.chain) + 1,
            "timestamp": time(),
            "transactions": self.current_transactions,
            "proof": proof,
            "previous_hash": previous_hash or self.hash(self.chain[-1]),
        }

        self.current_transactions = []
        self.chain.append(block)

        return block

    def new_transaction(self, **kwargs):
        self.current_transactions.append(kwargs)
        return self.last_block["index"] + 1

    @property
    def last_block(self):
        return self.chain[-1]

    @staticmethod
    def hash(block):
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def proof_of_work(self, last_block):
        last_proof = last_block["proof"]
        last_hash = self.hash(last_block)

        proof = 0
        while not valid_proof(last_proof, proof, last_hash):
            proof += 1

        return proof

    async def async_proof_of_work(self, last_block):
        return await asyncio.to_thread(self.proof_of_work, last_block)
