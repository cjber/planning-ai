from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from planning_ai.common.utils import Paths
from planning_ai.llms.llm import GPT4o

with open(Paths.PROMPTS / "policy.txt", "r") as f:
    policy_template = f.read()


class Policy(BaseModel):
    """Return condensed details and their associated doc_ids"""

    detail: str
    doc_id: list[int]


class PolicyList(BaseModel):
    policies: list[Policy]


SLLM = GPT4o.with_structured_output(PolicyList, strict=True)


policy_prompt = ChatPromptTemplate([("system", policy_template)])
policy_chain = policy_prompt | SLLM


if __name__ == "__main__":
    test_policy = "Protecting open spaces"
    test_bullet = [
        "The response emphasizes the need to preserve greenfield land, which relates to protecting open spaces.",
        "The response notes that greenspace land should be preserved.",
        "The response emphasizes the need for creating more parks, which relates to protecting open spaces.",
    ]
    test_docids = [1, 13, 21]

    result = policy_chain.invoke(
        {"theme": "Climate Change", "policy": test_policy, "details": zipped}
    )
    print(result)
