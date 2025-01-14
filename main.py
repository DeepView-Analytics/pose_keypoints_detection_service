import os
from fastapi import FastAPI

import uvicorn
from contextlib import asynccontextmanager

from pose_keypoints_estimator_service.kafka_interaction.kafka_consumer import KafkaConsumerService
from pose_keypoints_estimator_service.kafka_interaction.kafka_producer import KafkaProducerService



# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', '127.0.0.1:9092')
KAFKA_TOPIC_REQUESTS = os.getenv('KAFKA_TOPIC_REQUESTS', 'person_detected_response')


# Initialize Kafka Producer and Consumer
producer = KafkaProducerService(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
consumer = KafkaConsumerService(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, topic=KAFKA_TOPIC_REQUESTS,producer=producer)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    await producer.start()
    await consumer.start()
    yield
    # Shutdown logic
    await producer.close()
    await consumer.consumer.stop()  

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Person Detection Service!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('FASTAPI_PORT', 8000)))
