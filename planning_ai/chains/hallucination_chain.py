from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from planning_ai.common.utils import Paths
from planning_ai.llms.llm import GPT4o

with open(Paths.PROMPTS / "hallucination.txt", "r") as f:
    reduce_template = f.read()


class HallucinationChecker(BaseModel):
    """Grade the summary based upon the above criteria."""

    score: int = Field(..., description="Score for the summary")
    explanation: str = Field(..., description="Explain your reasoning for the score")


SLLM = GPT4o.with_structured_output(HallucinationChecker, strict=True)

hallucination_prompt = ChatPromptTemplate([("system", reduce_template)])
hallucination_chain = hallucination_prompt | SLLM

if __name__ == "__main__":
    test_document = """
    The Local Plan proposes a mass development north-west of Cambridge despite marked growth
    in the last twenty years or so following the previous New Settlement Study. In this period,
    the major settlement of Cambourne has been created - now over the projected 3,000 homes and
    Papworth Everard has grown beyond recognition. This in itself is a matter of concern.
    """

    result = hallucination_chain.invoke(
        {
            "summary": "The author fully supports the plan due to the nuclear power plant.",
            "document": test_document,
        }
    )
    print(result)
