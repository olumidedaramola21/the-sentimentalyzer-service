# the-sentimentalyzer-service
A blueprint for serving heavy Machine Learning models (like Transformer) via an HTTP API.

## Overview
The Sentimentalyzer is a blueprint for serving heavy Machine Learning models (like Transformers) via an HTTP API.
Unlike standard web applications which are **IO bound** (waiting for databases), Ml services are **Compute bound** (matrix multiplication). This project explores why standard backend patterns fails when servving AI, and implements the **Dynamic Batching** architecture used by production systems like Nvidia Triton, Ray Serve, and vLLM.
The goal of this project is to transform a fragile, high-latency prototype into a robust, high-throughput inference engine.

## The Experiment
This repository will contain two distinct implementations to demonstrate for the "Infa Gap":

- The Naive Implementation (Baseline)
The naive implementation consists of a standard FastAPI route that invokes the model directly on each request, causing incoming requests to be handled sequentially or to contend for the **Global Interpreter Lock (GIl)**, which in turn leads to linearly increasing as concurrency grows, since the GPU or CPU  ends up processing one inference at a time and leaves a large portion of its available compute capacity idle and underutilizied.

- The High-Performance Implementation (Goal)
The high-performance implementation is built around a producer-consumer architecture, where the producer asynchronously accepts incoming HTTP request and enqueues them, while a consumer continuously pulls multiple pending request from the queue, batches them together into a single tensor, and executes on effecient inference pass, resulting in a significantly higher throughput, stable and predictable latency under load, and near-maximum utilization of the underlying hardware.

## Tech Stack
- Language: Python 3.9+
- API Framework: FastAPI + Uvicorn (ASGI server)
- ML Framewrok: Pytorch & HuggingFace Transformers
- Model: distilbert-base-uncased-finetuned-sst-2-english (Sentiment Analysis)
- Benchmarking: Locust.io

## Getting Started
1. CLone the repo 

```
git clone git@github.com:olumidedaramola21/the-sentimentalyzer-service.git
cd sentimentalyzer
```

2. Set up Virtual Environment

- Windows (Bash / Poweshell)
```bash
python -m venv venv
source venv/Scripts/activate
```

- Mac/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

3.  Install Dependencies
```
pip install -r requirements.txt
```

4.  Running the Naive Server
This starts the baseline server that we intend to break
```
uvicorn naive_server:app --reload
```
wait for the logs to show: INFO:naive_server:Model loaded!

## Contributing
Feel free to open issues if you find optimizations or want to discuss.
