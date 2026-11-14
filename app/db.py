import json
from contextlib import contextmanager
import psycopg
from psycopg.rows import dict_row
from app.config import get_settings

@contextmanager
def connection():
    url = get_settings().database_url
    if not url:
        raise RuntimeError("Database URL is not configured. Please set the DATABASE_URL environment variable.")
    with psycopg.connect(url, row_factory=dict_row , connect_timeout=60) as conn:
        yield conn

def audit(conn,actor,tenant,action,detail):
    conn.execute('INSERT INTO audit_events (actor, tenant, action, detail) VALUES (%s,%s,%s,%s)',(actor,tenant,action,json.dumps(detail)))
