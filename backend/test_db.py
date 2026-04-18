# filename: backend/test_db.py
import sqlite3

conn = sqlite3.connect("diagnoses.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM scans")
rows = cursor.fetchall()

print("\n--- DATABASE SAVED DATA ---")
for row in rows:
    print(f"Name: {row[1]} | Age: {row[2]} | AI Result: {row[3]} | Time: {row[5]}")

conn.close()