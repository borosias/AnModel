from __future__ import annotations

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
# DIVERSITY CONFIGURATION
# ============================================================================
@dataclass
class DiversityConfig:
    """
    Parameters for controlling the diversity of generated data.
    """
    # --- User Activity ---
    activity_mu: float = 0.0
    activity_sigma: float = 0.7

    min_events_per_user: int = 5
    max_events_multiplier: float = 5.0

    # --- User Activity Time Window ---
    min_activity_span_days: int = 14
    max_activity_span_days: int = 365

    # --- Snapshot / Target Window ---
    snapshot_horizon_days: int = 7  # window size for calculating will_purchase_next_7d
    snapshot_date: datetime | None = None  # if None, take the current moment

    # --- Promo windows (Variant A) ---
    promo_share: float = 0.2              # share of users with promo activation
    promo_window_days: int = 10           # length of the promo window before snapshot
    promo_purchase_boost: float = 2.5     # boost of purchase probability in the promo window
    promo_max_session_gap_days: int = 1   # maximum days between sessions in the promo window

    # --- Snapshot-aware bursts (Variant B) ---
    burst_share: float = 0.25             # share of users with a targeted chain in the last days
    burst_force_purchase_prob: float = 0.8  # probability of completing the chain with a purchase

    # --- Intervals between sessions ---
    min_days_between_sessions: int = 0
    max_days_between_sessions: int = 5

    # --- Intervals between events within a session (seconds) ---
    min_seconds_between_events: int = 10
    max_seconds_between_events: int = 900

    # --- Session Size ---
    session_size_lambda: float = 1 / 3


# ============================================================================
# USER COHORTS: user profiles for realistic behavior
# ============================================================================
@dataclass
class UserCohort:
    """User cohort profile."""
    name: str
    events_multiplier: float      # multiplier for base events
    span_days_range: tuple        # (min, max) activity days
    session_size_lambda: float    # lambda for session size
    purchase_boost: float         # purchase probability multiplier in Markov


USER_COHORTS = {
    "heavy": UserCohort(
        name="heavy",
        events_multiplier=3.0,
        span_days_range=(180, 365),
        session_size_lambda=1/5,   # large sessions
        purchase_boost=1.5,
    ),
    "medium": UserCohort(
        name="medium",
        events_multiplier=1.0,
        span_days_range=(60, 180),
        session_size_lambda=1/3,
        purchase_boost=1.0,
    ),
    "light": UserCohort(
        name="light",
        events_multiplier=0.4,
        span_days_range=(14, 60),
        session_size_lambda=1/2,
        purchase_boost=0.7,
    ),
    "one_time": UserCohort(
        name="one_time",
        events_multiplier=0.15,
        span_days_range=(1, 7),
        session_size_lambda=1/2,
        purchase_boost=0.5,
    ),
}

COHORT_DISTRIBUTION = ["heavy", "medium", "medium", "light", "light", "light", "one_time"]

def is_flag_user(user_id: str, seed: int, ratio: float, salt: str) -> bool:
    return (hash((user_id, seed, salt)) % 1000) < int(ratio * 1000)


def get_user_cohort(user_id: str, seed: int = 42) -> UserCohort:
    """Deterministically selects a cohort for a user."""
    # Use hash for determinism
    h = hash((user_id, seed)) % len(COHORT_DISTRIBUTION)
    cohort_name = COHORT_DISTRIBUTION[h]
    return USER_COHORTS[cohort_name]


# Default config — can be overridden via CLI or a separate file
DEFAULT_DIVERSITY = DiversityConfig()

# ============================================================================

ITEMS = [f"item_{i}" for i in range(1, 201)]
REGIONS = ["UA-30", "UA-40", "UA-50"]
EVENTS = ["page_view", "product_view", "add_to_cart", "purchase", "search", "click"]
SEARCH_QUERIES = ["shoes", "blanket", "phone", "charger", "jacket", "backpack", "watch"]

# -- Logging --
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("data-gen")

# Generate weights for items according to Zipf's law (popularity decreases)
# item_1 will be the most popular, item_200 — the rarest.
item_ranks = range(1, len(ITEMS) + 1)
# alpha=1.5 — strong skew (hits very popular), alpha=1.1 — smoother
zipf_weights = [1.0 / (r ** 1.5) for r in item_ranks]
total_weight = sum(zipf_weights)
ITEM_PROBS = [w / total_weight for w in zipf_weights]

# ============================================================================
# MARKOV TRANSITIONS: conditional probabilities of the next event
# ============================================================================
# Order of EVENTS: ["page_view", "product_view", "add_to_cart", "purchase", "search", "click"]
# Each row is the probability distribution of the next event after the given one
MARKOV_TRANSITIONS = {
    "page_view":     [0.15, 0.35, 0.10, 0.02, 0.18, 0.20],  # after page_view often product_view
    "product_view":  [0.08, 0.15, 0.35, 0.12, 0.10, 0.20],  # after product_view often add_to_cart
    "add_to_cart":   [0.05, 0.10, 0.15, 0.45, 0.05, 0.20],  # after add_to_cart high probability of purchase!
    "purchase":      [0.30, 0.25, 0.05, 0.02, 0.18, 0.20],  # after purchase — new cycle
    "search":        [0.10, 0.45, 0.10, 0.02, 0.13, 0.20],  # after search often product_view
    "click":         [0.15, 0.35, 0.15, 0.05, 0.10, 0.20],  # after click often product_view
}


# ----------------- Event factory -----------------
def make_event(
        user_id: str,
        session_id: str,
        event_type: str,
        ts: datetime | None = None,
) -> dict:
    item = random.choices(ITEMS, weights=ITEM_PROBS, k=1)[0]

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
    base_price = 1000.0 / (int(item.split('_')[1]) + 1)  # The higher the item number, the lower the "base" price (conditionally)
    price = round(random.uniform(base_price * 0.8, base_price * 1.2), 2)

    if event_type in ("product_view", "add_to_cart", "purchase", "click"):
        ev["properties"] = {"item_id": item, "price": max(10, price)}
    elif event_type == "search":
        ev["properties"] = {"query": random.choice(SEARCH_QUERIES)}
    return ev


def make_event_contextual(
        user_id: str,
        session_id: str,
        event_type: str,
        ts: datetime | None = None,
        current_item: str | None = None,
) -> dict:
    """
    Creating an event taking session context into account.

    If current_item is set and the event is related to a product (add_to_cart, purchase),
    use the same item with high probability — this creates realistic chains.
    """
    # Decide whether to use the current item or select a new one
    if current_item and event_type in ("add_to_cart", "purchase"):
        # 70% — continue with the same item, 30% — new
        if random.random() < 0.7:
            item = current_item
        else:
            item = random.choices(ITEMS, weights=ITEM_PROBS, k=1)[0]
    elif current_item and event_type == "click":
        # 50% — same item
        if random.random() < 0.5:
            item = current_item
        else:
            item = random.choices(ITEMS, weights=ITEM_PROBS, k=1)[0]
    else:
        item = random.choices(ITEMS, weights=ITEM_PROBS, k=1)[0]

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

    base_price = 1000.0 / (int(item.split('_')[1]) + 1)
    price = round(random.uniform(base_price * 0.8, base_price * 1.2), 2)

    if event_type in ("product_view", "add_to_cart", "purchase", "click"):
        ev["properties"] = {"item_id": item, "price": max(10, price)}
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
    Event generation for a single user with adjustable diversity.
    """
    # --- NEW: Get user cohort ---
    cohort = get_user_cohort(user_id)


    if cohort.name == "one_time":
        is_promo_user = False
        is_burst_user = False
    else:
        is_promo_user = is_flag_user(user_id, 42, diversity.promo_share, "promo")
        is_burst_user = is_flag_user(user_id, 42, diversity.burst_share, "burst")

    # --- 1. Individual intensity (taking cohort into account) ---
    activity_factor = random.lognormvariate(
        mu=diversity.activity_mu,
        sigma=diversity.activity_sigma
    )
    # Apply cohort multiplier
    target_events = int(events_per_user * activity_factor * cohort.events_multiplier)
    target_events = max(diversity.min_events_per_user, target_events)
    target_events = min(target_events, int(events_per_user * diversity.max_events_multiplier))

    # --- 2. Activity time window (taking cohort into account) ---
    # Bind generation to snapshot date (if set)
    now = diversity.snapshot_date or datetime.now(timezone.utc)
    year_ago = now - timedelta(days=365)

    # FIXED: safe span_days calculation
    span_min = max(diversity.min_activity_span_days, cohort.span_days_range[0])
    span_max = max(span_min, min(diversity.max_activity_span_days, cohort.span_days_range[1]))

    span_days = random.randint(span_min, span_max)

    latest_start = now - timedelta(days=span_days)
    if latest_start > year_ago:
        user_start = year_ago + timedelta(
            days=random.randint(0, max(1, int((latest_start - year_ago).days)))
        )
    else:
        user_start = year_ago
    user_end = min(user_start + timedelta(days=span_days), now)

    if user_end < now - timedelta(days=diversity.snapshot_horizon_days):
        return

    # Target window boundaries (e.g., last 7 days before snapshot)
    target_window_start = now - timedelta(days=diversity.snapshot_horizon_days)
    promo_window_start = now - timedelta(days=diversity.promo_window_days)

    current_ts = user_start

    prev_event_type = None
    current_item = None

    # --- 3. Session and event generation ---
    has_purchase_last_window = False
    i = 0
    while i < target_events:
        remaining = target_events - i
        # Use lambda from cohort
        session_size = min(
            remaining,
            max(1, int(random.expovariate(cohort.session_size_lambda)) + 1)
        )
        session_id = str(uuid.uuid4())

        if i > 0:
            jump_days = random.randint(
                diversity.min_days_between_sessions,
                diversity.max_days_between_sessions
            )
            jump_seconds = random.randint(0, 6 * 60 * 60)
            # Reduce the gap between sessions in the promo window
            if is_promo_user and current_ts >= promo_window_start:
                jump_days = min(jump_days, diversity.promo_max_session_gap_days)
            current_ts = current_ts + timedelta(days=jump_days, seconds=jump_seconds)
            prev_event_type = None
            current_item = None

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

            # --- Markov choice taking cohort purchase_boost into account ---
            if prev_event_type is not None and prev_event_type in MARKOV_TRANSITIONS:
                weights = list(MARKOV_TRANSITIONS[prev_event_type])
                # Boost purchase probability (index 3) for "buying" cohorts
                weights[3] *= cohort.purchase_boost
                # Additional boost in promo window
                if is_promo_user and current_ts >= promo_window_start:
                    weights[3] *= diversity.promo_purchase_boost
                # Normalize
                total = sum(weights)
                weights = [w / total for w in weights]
            else:
                weights = list(event_distribution)
                if is_promo_user and current_ts >= promo_window_start:
                    weights[3] *= diversity.promo_purchase_boost
                    total = sum(weights)
                    weights = [w / total for w in weights]

            event_type = random.choices(EVENTS, weights=weights, k=1)[0]

            if event_type == "purchase" and prev_event_type not in ("add_to_cart",):
                continue

            ev = make_event_contextual(
                user_id, session_id, event_type,
                ts=event_ts,
                current_item=current_item
            )

            prev_event_type = event_type
            if event_type in ("product_view", "add_to_cart", "purchase", "click"):
                current_item = ev["properties"].get("item_id")

            async with sem:
                ok = await post_event(session, url, ev, timeout=timeout)
                counters["sent"] += 1
                if ok:
                    counters["ok"] += 1
                else:
                    counters["err"] += 1

            # Record purchases in the target window
            if event_type == "purchase" and target_window_start <= event_ts <= now:
                has_purchase_last_window = True

            i += 1
            if i >= target_events:
                break

            if delay_between_events > 0:
                await asyncio.sleep(delay_between_events)

    # --- 4A. Promo window: additional activity before snapshot ---
    if is_promo_user:
        promo_start = max(user_start, now - timedelta(days=diversity.promo_window_days))
        # Number of additional sessions in the promo window
        extra_sessions = random.randint(1, 3)
        for _ in range(extra_sessions):
            session_id = str(uuid.uuid4())
            # Fit the session into the promo range
            base_ts = promo_start + timedelta(
                days=random.uniform(0, max(0.1, (now - promo_start).days)),
                seconds=random.randint(0, 12 * 60 * 60)
            )
            # Shortened intervals between events create density
            session_len = random.randint(3, 8)
            prev_event_type = None
            current_item = None
            current_ts = base_ts
            for _j in range(session_len):
                # Boost purchase stronger in promo
                if prev_event_type is not None and prev_event_type in MARKOV_TRANSITIONS:
                    weights = list(MARKOV_TRANSITIONS[prev_event_type])
                    weights[3] *= (cohort.purchase_boost * diversity.promo_purchase_boost)
                    tw = sum(weights)
                    weights = [w / tw for w in weights]
                else:
                    base_w = list(event_distribution)
                    base_w[3] *= diversity.promo_purchase_boost
                    tw = sum(base_w)
                    weights = [w / tw for w in base_w]

                event_type = random.choices(EVENTS, weights=weights, k=1)[0]

                if event_type == "purchase" and prev_event_type not in ("add_to_cart",):
                    continue

                # Denser events within an hour
                delta_sec = random.randint(
                    max(5, diversity.min_seconds_between_events // 2),
                    max(30, diversity.min_seconds_between_events)
                )
                current_ts = min(now, current_ts + timedelta(seconds=delta_sec))

                ev = make_event_contextual(
                    user_id, session_id, event_type,
                    ts=current_ts,
                    current_item=current_item
                )
                if event_type in ("product_view", "add_to_cart", "purchase", "click"):
                    current_item = ev["properties"].get("item_id")

                async with sem:
                    ok = await post_event(session, url, ev, timeout=timeout)
                    counters["sent"] += 1
                    if ok:
                        counters["ok"] += 1
                    else:
                        counters["err"] += 1

                if event_type == "purchase" and target_window_start <= current_ts <= now:
                    has_purchase_last_window = True

                if delay_between_events > 0:
                    await asyncio.sleep(delay_between_events)

    # --- 4B. Snapshot-aware burst: intentional chain in the last days ---
    if is_burst_user and (not has_purchase_last_window):
        session_id = str(uuid.uuid4())
        base_ts = (
            max(user_start, now - timedelta(days=diversity.snapshot_horizon_days))
            + timedelta(
                days=random.uniform(0, diversity.snapshot_horizon_days),
                seconds=random.randint(0, 6 * 60 * 60)
            )
        )
        current_ts = base_ts
        # Warm-up scenario
        chain = ["search", "product_view", "product_view", "add_to_cart"]
        for idx, et in enumerate(chain):
            current_ts = min(now, current_ts + timedelta(seconds=random.randint(15, 90)))
            ev = make_event_contextual(user_id, session_id, et, ts=current_ts)
            async with sem:
                ok = await post_event(session, url, ev, timeout=timeout)
                counters["sent"] += 1
                if ok:
                    counters["ok"] += 1
                else:
                    counters["err"] += 1
            if delay_between_events > 0:
                await asyncio.sleep(delay_between_events)

        # Finish with a purchase with high probability
        if random.random() < diversity.burst_force_purchase_prob and current_ts < now:
            current_ts = min(now, current_ts + timedelta(seconds=random.randint(10, 60)))
            ev = make_event_contextual(user_id, session_id, "purchase", ts=current_ts)
            async with sem:
                ok = await post_event(session, url, ev, timeout=timeout)
                counters["sent"] += 1
                if ok:
                    counters["ok"] += 1
                else:
                    counters["err"] += 1
            if target_window_start <= current_ts <= now:
                has_purchase_last_window = True

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
    # CHANGED: More aggressive distribution for demo
    # Was: [0.25, 0.25, 0.10, 0.02, 0.20, 0.18] (purchase = 0.02)
    # Now: purchase = 0.08 (8%), add_to_cart = 0.20 (20%)
    # Order: ["page_view", "product_view", "add_to_cart", "purchase", "search", "click"]
    event_distribution = [0.15, 0.25, 0.20, 0.08, 0.12, 0.20]
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

    # --- Diversity parameters ---
    p.add_argument("--activity-sigma", type=float, default=0.7,
                   help="Sigma for lognormal activity distribution (0=uniform, 0.7=moderate, 1.0=high)")
    p.add_argument("--min-span-days", type=int, default=14,
                   help="Min activity window per user (days)")
    p.add_argument("--max-span-days", type=int, default=365,
                   help="Max activity window per user (days)")
    p.add_argument("--max-session-gap-days", type=int, default=5,
                   help="Max days between sessions")

    # --- Snapshot / promo / bursts ---
    p.add_argument("--snapshot-horizon-days", type=int, default=7,
                   help="Target window size in days before snapshot")
    p.add_argument("--snapshot-date", type=str, default="",
                   help="ISO datetime for snapshot (UTC). If empty, use now.")

    p.add_argument("--promo-share", type=float, default=0.2,
                   help="Share of users to activate promo window near snapshot [0..1]")
    p.add_argument("--promo-window-days", type=int, default=10,
                   help="Promo window length before snapshot in days")
    p.add_argument("--promo-purchase-boost", type=float, default=2.5,
                   help="Purchase probability boost multiplier within promo window")
    p.add_argument("--promo-max-session-gap-days", type=int, default=1,
                   help="Max days between sessions within promo window")

    p.add_argument("--burst-share", type=float, default=0.25,
                   help="Share of users to receive a snapshot-aware intent chain")
    p.add_argument("--burst-force-purchase-prob", type=float, default=0.8,
                   help="Probability to finish intent chain with purchase")

    return p.parse_args()


def main():
    args = parse_args()

    # Parse snapshot date
    snap_dt = None
    if args.snapshot_date:
        try:
            # Try full ISO first
            snap_dt = datetime.fromisoformat(args.snapshot_date)
            if snap_dt.tzinfo is None:
                snap_dt = snap_dt.replace(tzinfo=timezone.utc)
        except Exception:
            try:
                # Try YYYY-MM-DD
                snap_dt = datetime.strptime(args.snapshot_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except Exception:
                logger.warning(f"Cannot parse --snapshot-date={args.snapshot_date}, using now()")
                snap_dt = None

    diversity = DiversityConfig(
        activity_sigma=args.activity_sigma,
        min_activity_span_days=args.min_span_days,
        max_activity_span_days=args.max_span_days,
        max_days_between_sessions=args.max_session_gap_days,
        snapshot_horizon_days=args.snapshot_horizon_days,
        snapshot_date=snap_dt,
        promo_share=args.promo_share,
        promo_window_days=args.promo_window_days,
        promo_purchase_boost=args.promo_purchase_boost,
        promo_max_session_gap_days=args.promo_max_session_gap_days,
        burst_share=args.burst_share,
        burst_force_purchase_prob=args.burst_force_purchase_prob,
    )

    logger.info(
        "Starting generator: events_per_user=%s concurrency=%s snapshot_horizon=%sd snapshot_date=%s promo_share=%.2f burst_share=%.2f",
        args.events_per_user, args.concurrency, args.snapshot_horizon_days,
        (diversity.snapshot_date.isoformat() if diversity.snapshot_date else "now"),
        diversity.promo_share, diversity.burst_share,
    )
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