CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    user_uid UUID NOT NULL UNIQUE,
    region TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO users (user_uid, region)
SELECT gen_random_uuid(), (ARRAY['UA-30', 'UA-40', 'UA-50'])[floor(random()*3)+1]
FROM generate_series(1, 1000);

CREATE EXTENSION IF NOT EXISTS pgcrypto;
