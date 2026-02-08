"""
Naive baseline implementation used to illustrate the performance limitations of per-request model infernce. 
Request are processed independently without batching or coordination,resulting in poor throughput and rising tail latency  under concurrent load.
"""

import time
import logging
import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("naive_server")

app = FastAPI(title="Sentimentalyzer (Naive Baseline)")

# using a CPU-friendly model for this demo
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

logger.info("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

logger.info("Model loaded!")


class SentimentRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    """uptime check"""
    return {"status": "ok"}


@app.post("/predict")
def predict(request: SentimentRequest):
    """
    The Naive Approach:
    1. Receive request
    2. Tokenize (CPU bound)
    3. Model Inference (Matrix bound)
    4. Return

    Flaw: This endpoint is intentionally synchronous. While FastAPI executes `def` endpoints in a threadpool, 
    concurrent request still contend for CPU resources during tokenization and scheduling, and batching opportunities are lost. 
    This results in poor throughput and raising tail latency under load.
    """
    start_time = time.time()

    inputs = tokenizer(request.text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    positive_score = probabilities[0][1].item()
    sentiment = "POSITIVE" if positive_score > 0.5 else "NEGATIVE"

    process_time = (time.time() - start_time) * 1000

    return {"sentiment": sentiment, "score": positive_score, "latency_ms": process_time}
