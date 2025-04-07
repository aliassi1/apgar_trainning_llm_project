# Import necessary libraries for database operations and LLM functionality
from langchain_community.tools import QuerySQLDatabaseTool
from database import query_book_db
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain.tools import tool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize database connection to SQLite database
db = SQLDatabase.from_uri("sqlite:///books.db")

# Create a tool for executing SQL queries
execute_query = QuerySQLDatabaseTool(db=db)

# Initialize Google's Gemini model for natural language to SQL conversion
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ["API_KEY"], 
    temperature=0.5  # Controls randomness in responses
)

# Create a chain that converts natural language to SQL queries
sql_chain = create_sql_query_chain(
    llm=llm,
    db=db    
)

# Define the database retrieval tool using the @tool decorator
@tool
def retrive_data_from_database(query):
    """
    This tool is used to retrive data from the database.
    the database has the following columns:
    bookID, title, authors, average_rating, language_code, num_pages, price, available_in_stock, year, genre, or description

    Args:
        str: The query to retrive data from the database in natural language.
    Returns:
        list:The data retrived from the database.
    """
    print('##################################################')
    print('using database agent')
    print('##################################################')

    # Convert natural language query to SQL using the LLM chain
    sql_response = sql_chain.invoke({"question": query})
    
    # Extract the SQL query from the response
    sql_query = sql_response.split("SQLQuery:")[1].strip()
    
    # Execute the SQL query and get results
    response = execute_query.invoke({"query": sql_query})
    
    # Print the response for debugging
    print(response)
    
    return response
