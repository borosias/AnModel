import psycopg2

def load_users():
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="postgres",
        host="localhost",
        port=5432
    )
    cur = conn.cursor()
    cur.execute("SELECT user_uid FROM users;")
    users = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return users