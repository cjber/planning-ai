from dotenv import load_dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

load_dotenv()

rate_limiter = InMemoryRateLimiter(
    requests_per_second=50,
    check_every_n_seconds=0.1,
)
LLM = ChatOpenAI(temperature=0, model="gpt-4o-mini", rate_limiter=rate_limiter)
