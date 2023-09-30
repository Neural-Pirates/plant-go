<p>
    <img src="./img/logo.png">
</p>


## About

**PlantGo**: Rediscover Nature, One Tap at a Time. Gamify outdoor exploration, snap plant pictures, identify with AI and earn blockchain rewards. We're not just a game – we're a green revolution!


## Installation

Python 3.9 and higher are recommended for this project.

```
pip install -R miner_requirements.txt
pip install -R webserver_requirements.txt
```

## Usage

The webserver and miner can be run locally using [`aiohttp`](https://docs.aiohttp.org/en/stable/).

```
$MINER_PORT=8080
py miner.py
```

A miner address is **crucial** for this project .

```
$MINER_ADDRESS="..."
py webserver.py $MINER_ADDRESS
```
