"""
db_connection.py
------------------
Reusable MySQL connection helper for the whole project.
Import get_db_connection() wherever you need to talk to MySQL.
"""

import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "traffic_accident_db")


def get_db_connection():
    """Create and return a new MySQL connection using .env credentials."""
    return mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
    )


if __name__ == "__main__":
    # quick test: run `python database/db_connection.py` to check the connection works
    try:
        conn = get_db_connection()
        print("Connected to MySQL successfully!")
        conn.close()
    except Exception as e:
        print(f"Connection failed: {e}")