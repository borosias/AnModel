import asyncio
import json
from pydantic import BaseModel
from fastapi import FastAPI, Response, status
from aiokafka import AIOKafkaProducer

class Event(BaseModel):
    event_id: str
    event_type: str
    timestamp: str
    session_id: str
    user_id: str
    region: str
    properties: dict = {}

app = FastAPI()
producer = None
KAFKA_BOOTSTRAP = "kafka:9092"
TOPIC = "raw-events"

@app.on_event("startup")
async def startup_event():
    global producer
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP)
    await producer.start()

@app.on_event("shutdown")
async def shutdown_event():
    global producer
    if producer:
        await producer.stop()

@app.post("/collect", status_code=201)
async def collect(event: Event):
    payload = json.dumps(event.dict(), ensure_ascii=False).encode("utf-8")
    await producer.send_and_wait(TOPIC, payload)
    return Response(status_code=status.HTTP_201_CREATED)
