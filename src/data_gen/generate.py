import uuid
import random
import time
import requests
from datetime import datetime, timezone

URL = "http://localhost:8000/collect"
ITEMS = [f"item_{i}" for i in range(1, 201)]
REGIONS = ["UA-30", "UA-40", "UA-50"]
EVENTS = ["page_view", "product_view", "add_to_cart", "purchase", "search", "click"]

def make_event(user_id: str):
    event_type = random.choice(EVENTS)
    item = random.choice(ITEMS)
    ev = {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": str(uuid.uuid4()),
        "user_id": user_id,
        "region": random.choice(REGIONS),
        "properties": {}
    }
    if event_type in ("product_view", "add_to_cart", "purchase", "click"):
        ev["properties"] = {"item_id": item, "price": round(random.uniform(10, 500),2)}
    if event_type == "search":
        ev["properties"] = {"query": random.choice(["shoes","blanket","phone","charger"])}
    return ev

def send_batch(n_users=100, events_per_user=50, delay=0.01):
    for _ in range(n_users):
        user = f"user_{random.randint(1,100000)}"
        for _ in range(events_per_user):
            ev = make_event(user)
            try:
                requests.post(URL, json=ev, timeout=1)
            except Exception:
                pass
            time.sleep(delay)

if __name__ == "__main__":
    send_batch(200, 200, 0.005)
