from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from planning_ai.common.utils import Paths
from planning_ai.llms.llm import LLM

with open(Paths.PROMPTS / "policy.txt", "r") as f:
    policy_template = f.read()


policy_prompt = ChatPromptTemplate([("system", policy_template)])
policy_chain = policy_prompt | LLM | StrOutputParser()


if __name__ == "__main__":
    test_policy = "Protecting open spaces"
    test_bullet = "* " + "\n* ".join(
        [
            "The response emphasizes the need to preserve greenfield land, which relates to protecting open spaces.",
            "The response notes that greenspace land should be preserved."
            "The response emphasizes the need for creating more parks, which relates to protecting open spaces.",
        ]
    )

    result = policy_chain.invoke({"policy": test_policy, "bullet_points": test_bullet})
    print(result)
