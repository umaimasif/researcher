import sqlite3

API_KEY = "sk-12345-secret"

def get_user(name):
    conn = sqlite3.connect("db.sqlite")
    query = "SELECT * FROM users WHERE name = '" + name + "'"
    return conn.execute(query).fetchall()
