import sqlite3
import pandas as pd
from langchain.schema import Document

df=pd.read_csv("books.csv")

conn = sqlite3.connect("books.db")

# Save the DataFrame to a table named "books"
df.to_sql("books", conn, if_exists="replace", index=False)

conn.close()



def query_book_db(query: str) -> str:
    conn = sqlite3.connect("books.db")
    cursor = conn.cursor()

    result = cursor.execute(f"""
        SELECT title, authors, price, available_in_stock 
        FROM books 
        WHERE title LIKE '%' || ? || '%' OR authors LIKE '%' || ? || '%'
    """, (query, query)).fetchall()

    conn.close()

    if not result:
        return "Sorry, no matching books found."

    response = ""
    for row in result:
        response += (
            f"📘 Title: {row[0]}\n"
            f"✍️ Author: {row[1]}\n"
            f"💵 Price: ${row[2]}\n"
            f"📦 Stock: {row[3]}\n\n"
        )
    return response.strip()

