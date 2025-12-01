import argparse
import asyncio
import json
import logging
import random
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import List

import aiohttp

from postgres_load import load_users

# -- Конфигурация списка сущностей (не меняем формат полей!) --
ITEMS = [f"item_{i}" for i in range(1, 201)]
REGIONS = ["UA-30", "UA-40", "UA-50"]
EVENTS = ["page_view", "product_view", "add_to_cart", "purchase", "search", "click"]
SEARCH_QUERIES = ["shoes", "blanket", "phone", "charger", "jacket", "backpack", "watch"]

# -- Логирование --
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("data-gen")

# ----------------- Event factory (строго сохраняет структуру) -----------------
def make_event(user_id: str, session_id: str, event_type: str) -> dict:
    item = random.choice(ITEMS)
    # Randomize timestamp within the last ~1 year (365 days)
    now = datetime.now(timezone.utc)
    year_ago = now - timedelta(days=365)
    # total seconds between now and year_ago; choose a random offset
    total_sec = int((now - year_ago).total_seconds())
    rand_sec = random.randint(0, total_sec)
    ts = (year_ago + timedelta(seconds=rand_sec)).isoformat()
    ev = {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "timestamp": ts,
        "session_id": session_id,
        "user_id": user_id,
        "region": random.choice(REGIONS),
        "properties": {}
    }
    if event_type in ("product_view", "add_to_cart", "purchase", "click"):
        ev["properties"] = {"item_id": item, "price": round(random.uniform(10, 500), 2)}
    elif event_type == "search":
        ev["properties"] = {"query": random.choice(SEARCH_QUERIES)}
    return ev

# ----------------- Async sender with retries -----------------
async def post_event(session: aiohttp.ClientSession, url: str, payload: dict, timeout: float, retries: int = 2) -> bool:
    backoff = 0.5
    for attempt in range(retries + 1):
        try:
            # Use json= to let aiohttp handle encoding and headers
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status in (200, 201, 202):
                    return True
                # treat 4xx/5xx as error to retry (except some 4xx maybe)
                text = await resp.text()
                logger.debug(f"bad status {resp.status}: {text}")
        except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionResetError, OSError) as ex:
            # WinError 64 (network name no longer available) maps to OSError on Windows
            logger.debug(f"request error (will retry if attempts left): {ex}")
        if attempt < retries:
            await asyncio.sleep(backoff)
            backoff *= 2
    return False

# ----------------- Worker that produces events for one user -----------------
async def produce_for_user(
    user_id: str,
    events_per_user: int,
    url: str,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    delay_between_events: float,
    seq_session_prob: float,
    event_distribution: List[float],
    timeout: float,
    counters: dict,
):
    # create several sessions per user: group events into chunks
    i = 0
    while i < events_per_user:
        # decide session length (1..min(remaining, 20))
        remaining = events_per_user - i
        # geometric-like session size biased to small sessions
        session_size = min(remaining, max(1, int(random.expovariate(1/3)) + 1))
        session_id = str(uuid.uuid4())
        for _ in range(session_size):
            event_type = random.choices(EVENTS, weights=event_distribution, k=1)[0]
            ev = make_event(user_id, session_id, event_type)
            async with sem:
                ok = await post_event(session, url, ev, timeout=timeout)
                counters["sent"] += 1
                if ok:
                    counters["ok"] += 1
                else:
                    counters["err"] += 1
            i += 1
            if delay_between_events > 0:
                await asyncio.sleep(delay_between_events)
    return

# ----------------- Main orchestrator -----------------
async def run_generation(
    url: str,
    events_per_user: int,
    concurrency: int,
    delay_between_events: float,
    timeout: float,
    seed: int,
):
    random.seed(seed)
    # realistic event distribution: more views/clicks, fewer purchases
    # weights correspond to EVENTS order: ["page_view","product_view","add_to_cart","purchase","search","click"]
    event_distribution = [0.25, 0.25, 0.10, 0.02, 0.20, 0.18]

    sem = asyncio.Semaphore(concurrency)
    # Respect provided timeout for read operations; leave total unlimited to allow streaming batches
    timeout_obj = aiohttp.ClientTimeout(total=None, sock_connect=5, sock_read=timeout)

    counters = {"sent": 0, "ok": 0, "err": 0}
    start_t = time.time()

    # enable_cleanup_closed helps avoid ConnectionResetError on Windows when peers close abruptly
    conn = aiohttp.TCPConnector(limit=concurrency * 2, force_close=False, enable_cleanup_closed=True, ttl_dns_cache=10)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout_obj) as session:
        tasks = []
        user_ids = load_users()
        for user in user_ids:
            t = asyncio.create_task(
                produce_for_user(
                    user,
                    events_per_user,
                    url,
                    session,
                    sem,
                    delay_between_events,
                    seq_session_prob=0.6,
                    event_distribution=event_distribution,
                    timeout=timeout,
                    counters=counters,
                )
            )
            tasks.append(t)
            # throttle creation of tasks to avoid burst memory use
            if len(tasks) >= concurrency * 4:
                _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                # asyncio.wait returns sets; keep pending tasks and continue adding to a list
                tasks = list(pending)
        if tasks:
            await asyncio.gather(*tasks)

    elapsed = time.time() - start_t
    logger.info(f"Generation finished. sent={counters['sent']} ok={counters['ok']} err={counters['err']} elapsed={elapsed:.2f}s rps={counters['sent']/elapsed:.2f}")

# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser(description="High-throughput event generator (compatible with schema)")
    p.add_argument("--url", default="http://localhost:8000/collect", help="Collector URL")
    p.add_argument("--events-per-user", type=int, default=200, help="Events per user")
    p.add_argument("--concurrency", type=int, default=100, help="Concurrent inflight HTTP requests")
    p.add_argument("--delay", type=float, default=0.0, help="Delay between events per user (seconds)")
    p.add_argument("--timeout", type=float, default=3.0, help="HTTP timeout (seconds)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()

def main():
    args = parse_args()
    logger.info(f"Starting generator: events_per_user={args.events_per_user} concurrency={args.concurrency} delay={args.delay}")
    asyncio.run(run_generation(args.url, args.events_per_user, args.concurrency, args.delay, args.timeout, args.seed))

if __name__ == "__main__":
    main()
