from typing import Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator

from planning_ai.llms.llm import LLM

with open("./planning_ai/chains/prompts/map.txt", "r") as f:
    map_template = f.read()


class BriefSummary(BaseModel):
    """A summary of the response with generated metadata"""

    summary: str = Field(..., description="Summary of the response.")
    stance: Literal["SUPPORT", "OPPOSE", "NEUTRAL"] = Field(
        ..., description="Overall stance of the response."
    )
    themes: list[str] = Field(
        ..., description="A list of themes associated with the response."
    )
    rating: int = Field(
        ...,
        description="How constructive the response is, from a rating of 1 to 10.",
    )

    @validator("summary")
    def summary_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Summary cannot be empty.")
        return v


SLLM = LLM.with_structured_output(BriefSummary)


map_prompt = ChatPromptTemplate.from_messages([("system", map_template)])
map_chain = map_prompt | SLLM

if __name__ == "__main__":
    test_document = """
    The Local Plan proposes a mass development north-west of Cambridge despite marked growth
    in the last twenty years or so following the previous New Settlement Study. In this period,
    the major settlement of Cambourne has been created - now over the projected 3,000 homes and
    Papworth Everard has grown beyond recognition. This in itself is a matter of concern.
    """

    result = map_chain.invoke({"context": test_document})

    print("Generated Summary:")
    print(result)
    result
