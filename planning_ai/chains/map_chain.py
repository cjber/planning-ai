from enum import Enum

from langchain.output_parsers import RetryOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from planning_ai.common.utils import Paths
from planning_ai.llms.llm import LLM
from planning_ai.themes import PolicySelection, Theme

with open(Paths.PROMPTS / "themes.txt", "r") as f:
    themes_txt = f.read()

with open(Paths.PROMPTS / "map.txt", "r") as f:
    map_template = f"{themes_txt}\n\n {f.read()}"


class Sentiment(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Place(BaseModel):
    """Represents a geographical location mentioned in the response with associated sentiment."""

    place: str = Field(
        ...,
        description=(
            "The name of the geographical location mentioned in the response. "
            "This can be a city, town, region, or any identifiable place."
        ),
    )
    sentiment: Sentiment = Field(
        ...,
        description=(
            "The sentiment associated with the mentioned place, categorized as 'positive', 'negative', or 'neutral'. "
            "Assess sentiment based on the context in which the place is mentioned, considering both positive and negative connotations."
        ),
    )


class BriefSummary(BaseModel):
    """A summary of the response with generated metadata"""

    summary: str = Field(
        ...,
        description=(
            "A concise summary of the response, capturing the main points and overall sentiment. "
            "The summary should reflect the key arguments and conclusions presented in the response."
        ),
    )
    themes: list[Theme] = Field(
        ...,
        description=(
            "A list of themes associated with the response. Themes are overarching topics or "
            "categories that the response addresses, such as 'Climate change' or 'Infrastructure'. "
            "Identify themes based on the content and context of the response."
        ),
    )
    policies: list[PolicySelection] = Field(
        ...,
        description=(
            "A list of policies associated with the response, each accompanied by directly related "
            "information as bullet points. Bullet points should provide specific details or examples "
            "that illustrate how the policy is relevant to the response."
        ),
    )
    places: list[Place] = Field(
        ...,
        description=(
            "All places mentioned in the response, with the sentiment categorized as 'positive', 'negative', or 'neutral'. "
            "A place can be a city, region, or any geographical location. Assess sentiment based on the context "
            "in which the place is mentioned, considering both positive and negative connotations."
        ),
    )
    is_constructive: bool = Field(
        ...,
        description=(
            "A flag indicating whether the response is constructive. A response is considered constructive if it "
            "provides actionable suggestions or feedback, addresses specific themes or policies, and is presented "
            "in a coherent and logical manner."
        ),
    )


SLLM = LLM.with_structured_output(BriefSummary, strict=False)

# TODO: Split out the policy stuff from this class. Find policies later based on
# what themes are already identified (should improve accuracy)
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
    __import__("pprint").pprint(dict(result))
