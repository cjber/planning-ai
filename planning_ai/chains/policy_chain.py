from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from planning_ai.common.utils import Paths
from planning_ai.llms.llm import LLM

with open(Paths.PROMPTS / "policy.txt", "r") as f:
    policy_template = f.read()


class PolicyMerger(BaseModel):
    """Return condensed details and their associated doc_ids"""

    details: list[str]
    doc_id: list[list[int]]


SLLM = LLM.with_structured_output(PolicyMerger, strict=True)


policy_prompt = ChatPromptTemplate([("system", policy_template)])
policy_chain = policy_prompt | SLLM


if __name__ == "__main__":
    test_policy = "Protecting open spaces"
    test_bullet = """
* The response emphasizes the need to preserve greenfield land, which relates to protecting open spaces [1].\n
* The response notes that greenspace land should be preserved [13].\n
* The response emphasizes the need for creating more parks, which relates to protecting open spaces [21].
            """

    result = policy_chain.invoke({"policy": test_policy, "bullet_points": test_bullet})
    print(result)
