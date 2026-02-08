from locust import HttpUser, task, between
import random

# sample reviews to send
REVIEWS = [
    "The product was amazing! I loved the interface and the speed.",
    "Terriblr experience. The app crashed three times and support was rude.",
    "It's okay, not great but gets the job done. Average performance.",
    "Absolutely fantastic service, would recommend to everyone I know",
    "I hate it. It's the worst thing i've ever bought.",
]


class SentimentUser(HttpUser):

    wait_time = between(0.1, 0.5)  # Aggressive user behavior

    @task
    def predict_sentiment(self):
        text = random.choice(REVIEWS)

        with self.client.post(
            "/predict", json={"text": text}, catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()

                if "latency_ms" not in data:
                    response.failure("Response missing latency_ms")
            else:
                response.failure(f"Status code: {response.status_code}")
