from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

LLM = ChatOpenAI(temperature=0, model="gpt-4o")
