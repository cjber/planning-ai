from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

GPT4o = ChatOpenAI(temperature=0, model="gpt-4o-mini")
O3Mini = ChatOpenAI(model="o3-mini")
