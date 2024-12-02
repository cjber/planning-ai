from enum import Enum
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from planning_ai.common.utils import Paths
from planning_ai.llms.llm import LLM

with open(Paths.PROMPTS / "map.txt", "r") as f:
    map_template = f.read()


class Theme(str, Enum):
    climate = "Climate change"
    biodiversity = "Biodiversity and green spaces"
    wellbeing = "Wellbeing and social inclusion"
    great_places = "Great places"
    jobs = "Jobs"
    homes = "Homes"
    infrastructure = "Infrastructure"

    def __repr__(self) -> str:
        return self.value


class Place(BaseModel):
    place: str = Field(..., description="Place mentioned in the response.")
    sentiment: int = Field(..., description="Related sentiment ranked 1 to 10.")


class BriefSummary(BaseModel):
    """A summary of the response with generated metadata"""

    summary: str = Field(..., description="A summary of the response.")
    stance: Literal["SUPPORT", "OPPOSE", "MIXED", "NEUTRAL"] = Field(
        ...,
        description="Overall stance of the response. Either SUPPORT, OPPOSE, MIXED, or NEUTRAL.",
    )
    themes: list[Theme] = Field(
        ..., description="A list of themes associated with the response."
    )
    places: list[Place] = Field(
        ...,
        description="All places mentioned in the response, with the positivity of the related sentiment ranked 1 to 10",
    )
    rating: int = Field(
        ...,
        description="How constructive the response is, from a rating of 1 to 10.",
    )

    def __str__(self) -> str:
        return (f"Summary:\n\n{self.summary}\n\n" "Related Themes:\n\n") + "\n".join(
            [f"{idx+1}: {theme}" for (idx, theme) in enumerate(self.themes)]
        )


SLLM = LLM.with_structured_output(BriefSummary, strict=True)

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
    print(result)
