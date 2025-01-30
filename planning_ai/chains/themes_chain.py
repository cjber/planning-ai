from enum import Enum
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from planning_ai.common.utils import Paths
from planning_ai.llms.llm import LLM


class Theme(Enum):
    climate_change = "Climate Change"
    biodiversity = "Biodiversity and Green Spaces"
    wellbeing = "Wellbeing and Social Inclusion"
    great_places = "Great Places"
    jobs = "Jobs"
    homes = "Homes"
    infrastructure = "Infrastructure"


class ThemeScore(BaseModel):
    theme: Theme
    score: int


class ThemeSelector(BaseModel):
    themes: Optional[list[ThemeScore]]


with open(Paths.PROMPTS / "themes.txt", "r") as f:
    themes_template = f.read()

themes_prompt = ChatPromptTemplate.from_messages([("system", themes_template)])

SLLM = LLM.with_structured_output(ThemeSelector, strict=True)

themes_chain = themes_prompt | SLLM


if __name__ == "__main__":
    test_document = """
    The Local Plan proposes a mass development north-west of Cambridge despite marked growth
    in the last twenty years or so following the previous New Settlement Study. In this period,
    the major settlement of Cambourne has been created - now over the projected 3,000 homes and
    Papworth Everard has grown beyond recognition. This in itself is a matter of concern.
    """

    result = themes_chain.invoke({"document": test_document})
    __import__("pprint").pprint(dict(result))
