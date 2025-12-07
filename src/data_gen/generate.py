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
from dataclasses import dataclass

import aiohttp

from postgres_load import load_users


# ============================================================================
# КОНФИГУРАЦИЯ РАЗНООБРАЗИЯ
# ============================================================================
@dataclass
class DiversityConfig:
    """
    Параметры для управления разнообразием генерируемых данных.

    Все параметры можно менять без изменения логики — просто подбирай значения.
    """

    # --- Активность пользователей ---
    # Логнормальное распределение: activity = base * exp(N(mu, sigma))
    # mu=0, sigma=0 → все одинаковые
    # mu=0, sigma=0.7 → разброс ~0.3x–3x (умеренный)
    # mu=0, sigma=1.0 → разброс ~0.1x–7x (сильный)
    activity_mu: float = 0.0
    activity_sigma: float = 0.7

    # Ограничения на итоговое количество событий
    min_events_per_user: int = 5
    max_events_multiplier: float = 5.0  # макс = base * multiplier

    # --- Временное окно активности пользователя ---
    # Каждый юзер активен в своём "окне" внутри последнего года
    min_activity_span_days: int = 14  # минимум дней активности
    max_activity_span_days: int = 365  # максимум

    # --- Интервалы между сессиями ---
    min_days_between_sessions: int = 0
    max_days_between_sessions: int = 5

    # --- Интервалы между событиями внутри сессии (секунды) ---
    min_seconds_between_events: int = 10
    max_seconds_between_events: int = 900  # 15 минут

    # --- Размер сессий ---
    # Используется экспоненциальное распределение: avg_session_size = 1/lambda
    session_size_lambda: float = 1 / 3  # в среднем ~3 события на сессию


# Дефолтный конфиг — можно переопределить через CLI или отдельный файл
DEFAULT_DIVERSITY = DiversityConfig()

# ============================================================================

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


# ----------------- Event factory -----------------
def make_event(
        user_id: str,
        session_id: str,
        event_type: str,
        ts: datetime | None = None,
) -> dict:
    item = random.choice(ITEMS)

    if ts is None:
        now = datetime.now(timezone.utc)
        year_ago = now - timedelta(days=365)
        total_sec = int((now - year_ago).total_seconds())
        rand_sec = random.randint(0, total_sec)
        ts = year_ago + timedelta(seconds=rand_sec)

    ev = {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "timestamp": ts.isoformat(),
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
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status in (200, 201, 202):
                    return True
                text = await resp.text()
                logger.debug(f"bad status {resp.status}: {text}")
        except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionResetError, OSError) as ex:
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
        diversity: DiversityConfig,
):
    """
    Генерация событий для одного пользователя с настраиваемым разнообразием.
    """
    # --- 1. Индивидуальная интенсивность ---
    activity_factor = random.lognormvariate(
        mu=diversity.activity_mu,
        sigma=diversity.activity_sigma
    )
    target_events = int(events_per_user * activity_factor)
    target_events = max(diversity.min_events_per_user, target_events)
    target_events = min(target_events, int(events_per_user * diversity.max_events_multiplier))

    # --- 2. Временное окно активности ---
    now = datetime.now(timezone.utc)
    year_ago = now - timedelta(days=365)

    span_days = random.randint(
        diversity.min_activity_span_days,
        diversity.max_activity_span_days
    )

    latest_start = now - timedelta(days=span_days)
    if latest_start > year_ago:
        user_start = year_ago + timedelta(
            days=random.randint(0, max(1, int((latest_start - year_ago).days)))
        )
    else:
        user_start = year_ago
    user_end = min(user_start + timedelta(days=span_days), now)

    current_ts = user_start

    # --- 3. Генерация сессий и событий ---
    i = 0
    while i < target_events:
        remaining = target_events - i
        session_size = min(
            remaining,
            max(1, int(random.expovariate(diversity.session_size_lambda)) + 1)
        )
        session_id = str(uuid.uuid4())

        # Между сессиями
        if i > 0:
            jump_days = random.randint(
                diversity.min_days_between_sessions,
                diversity.max_days_between_sessions
            )
            jump_seconds = random.randint(0, 6 * 60 * 60)
            current_ts = current_ts + timedelta(days=jump_days, seconds=jump_seconds)

        if current_ts > user_end:
            current_ts = user_end - timedelta(hours=1)

        for _ in range(session_size):
            event_ts = current_ts + timedelta(
                seconds=random.randint(
                    diversity.min_seconds_between_events,
                    diversity.max_seconds_between_events
                )
            )
            current_ts = event_ts

            if current_ts > user_end:
                current_ts = user_end

            event_type = random.choices(EVENTS, weights=event_distribution, k=1)[0]
            ev = make_event(user_id, session_id, event_type, ts=event_ts)

            async with sem:
                ok = await post_event(session, url, ev, timeout=timeout)
                counters["sent"] += 1
                if ok:
                    counters["ok"] += 1
                else:
                    counters["err"] += 1

            i += 1
            if i >= target_events:
                break

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
        diversity: DiversityConfig,
):
    random.seed(seed)
    event_distribution = [0.25, 0.25, 0.10, 0.02, 0.20, 0.18]

    sem = asyncio.Semaphore(concurrency)
    timeout_obj = aiohttp.ClientTimeout(total=None, sock_connect=5, sock_read=timeout)

    counters = {"sent": 0, "ok": 0, "err": 0}
    start_t = time.time()

    conn = aiohttp.TCPConnector(limit=concurrency * 2, force_close=False, enable_cleanup_closed=True, ttl_dns_cache=10)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout_obj) as session:
        tasks = []
        user_ids = load_users()

        logger.info(f"Users to process: {len(user_ids)}")
        logger.info(f"Base events per user: {events_per_user}")
        logger.info(
            f"Diversity config: sigma={diversity.activity_sigma}, span={diversity.min_activity_span_days}-{diversity.max_activity_span_days}d")

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
                    diversity=diversity,
                )
            )
            tasks.append(t)
            if len(tasks) >= concurrency * 4:
                _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = list(pending)
        if tasks:
            await asyncio.gather(*tasks)

    elapsed = time.time() - start_t
    logger.info(
        f"Generation finished. sent={counters['sent']} ok={counters['ok']} err={counters['err']} elapsed={elapsed:.2f}s rps={counters['sent'] / elapsed:.2f}")


# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser(description="High-throughput event generator with diversity control")
    p.add_argument("--url", default="http://localhost:8000/collect", help="Collector URL")
    p.add_argument("--events-per-user", type=int, default=200, help="Base events per user")
    p.add_argument("--concurrency", type=int, default=100, help="Concurrent HTTP requests")
    p.add_argument("--delay", type=float, default=0.0, help="Delay between events (seconds)")
    p.add_argument("--timeout", type=float, default=3.0, help="HTTP timeout (seconds)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # --- Параметры разнообразия ---
    p.add_argument("--activity-sigma", type=float, default=0.7,
                   help="Sigma for lognormal activity distribution (0=uniform, 0.7=moderate, 1.0=high)")
    p.add_argument("--min-span-days", type=int, default=14,
                   help="Min activity window per user (days)")
    p.add_argument("--max-span-days", type=int, default=365,
                   help="Max activity window per user (days)")
    p.add_argument("--max-session-gap-days", type=int, default=5,
                   help="Max days between sessions")

    return p.parse_args()


def main():
    args = parse_args()

    diversity = DiversityConfig(
        activity_sigma=args.activity_sigma,
        min_activity_span_days=args.min_span_days,
        max_activity_span_days=args.max_span_days,
        max_days_between_sessions=args.max_session_gap_days,
    )

    logger.info(f"Starting generator: events_per_user={args.events_per_user} concurrency={args.concurrency}")
    asyncio.run(run_generation(
        args.url,
        args.events_per_user,
        args.concurrency,
        args.delay,
        args.timeout,
        args.seed,
        diversity,
    ))


if __name__ == "__main__":
    main()