CREATE DATABASE IF NOT EXISTS default;

CREATE TABLE IF NOT EXISTS events_raw
(
    event_id String,
    event_type String,
    ts DateTime,
    session_id String,
    user_id String,
    region String,
    item_id String,
    price Nullable(Float64),
    properties String
)
ENGINE = MergeTree()
ORDER BY (ts);
