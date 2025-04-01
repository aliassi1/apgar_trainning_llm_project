# Import required libraries
import os 
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import create_react_agent,AgentExecutor  
from tools.retrive_data_from_database import retrive_data_from_database
from tools.document_ret import document_ret
from langchain import hub
from langchain.memory import ConversationBufferMemory

# Load environment variables from .env file
load_dotenv()

# Initialize the Google Generative AI model (Gemini)
llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            api_key=os.environ["API_KEY"], 
            temperature=0.5  # Controls randomness in responses (0.0 = deterministic, 1.0 = most random)
        )

# Set up conversation memory to maintain context between queries
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Define the initial user query
message="which lord of the rings book is available?"
# Alternative query (commented out)
# message="what is the return policy for the company?"

# Load the ReAct prompt template from LangChain hub
prompt = hub.pull("hwchase17/react")

# Create an agent with the LLM and two tools: database retrieval and document retrieval
agent = create_react_agent(llm, [retrive_data_from_database, document_ret],prompt)

# Set up the agent executor with memory capability
agent_executor = AgentExecutor(
    agent=agent, 
    tools=[retrive_data_from_database, document_ret], 
    verbose=False,
    memory=memory
)

# First query execution
try:
    # Execute the first query about LOTR books
    response=agent_executor.invoke(
        { 
        "input": message,
    }
    )
    print("Agent Response:", response["output"])

except Exception as e:
    print("Error during agent execution:", str(e))

# Second query execution
try:
    # Follow-up question about availability and prices
    message="how many are available from the ones you gave me and the price of each?"
    response=agent_executor.invoke(
        { 
        "input": message,
    }
    )
    print("Agent Response:", response["output"])

except Exception as e:
    print("Error during agent execution:", str(e))

# Commented out classifier code that could be used for query intent classification
# Classifier_prompt = [
#     {
#         "role": "system",
#         "content": (
#             "You are an intent classifier. Classify user queries into one of the following types:\n"
#         "- 'database' if the query is about structured fields like bookID, title, authors, average_rating, language_code, num_pages, price, available_in_stock, year, genre, or description.\n"
#         "- 'RAG': if the answer is based on policy documents.\n"
#         "- 'general knowledge' if the query asks about external facts unrelated to the database or stored documents.\n"
#         "Respond with only one word: database, RAG, or general knowledge."
#     )
#     }]

# Classifier_prompt.append({"role": "user", "content": message})
# print(response['result'])
# print('this is the source document',response['source_documents'][0].metadata['source'])


