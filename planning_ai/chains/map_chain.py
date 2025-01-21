from enum import Enum, auto
from typing import Optional, Set, Type

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, create_model

from planning_ai.common.utils import Paths
from planning_ai.llms.llm import LLM
from planning_ai.themes import THEMES_AND_POLICIES

with open(Paths.PROMPTS / "map.txt", "r") as f:
    map_template = f.read()


def create_policy_enum(
    policy_groups: Set[str], name: str = "DynamicPolicyEnum"
) -> Enum:
    """
    Create a dynamic enum for policies based on the given policy groups.

    Args:
        policy_groups (Set[str]): A set of policy group names.
        name (str): Name of the enum to be created.

    Returns:
        Type[Enum]: A dynamically created Enum class for the policies.
    """
    return Enum(name, {policy: auto() for policy in policy_groups})


def create_brief_summary_model(policy_enum: Enum) -> Type[BaseModel]:
    """
    Dynamically create a BriefSummary model using the provided policy enum.

    Args:
        policy_enum (Type[Enum]): The dynamically created policy enum.

    Returns:
        Type[BaseModel]: A dynamically generated Pydantic model for BriefSummary.
    """

    # NOTE: For some reason GPT4o doesn't work if we use too much structure
    DynamicPolicy = create_model(
        "DynamicPolicy",
        # policy=(policy_enum, ...),
        policy=(str, ...),
        note=(str, ...),
        __config__={"extra": "forbid"},
    )

    return create_model(
        "DynamicBriefSummary",
        summary=(str, ...),
        policies=(Optional[list[DynamicPolicy]], ...),
        __module__=__name__,
        __config__={"extra": "forbid"},
    )


def create_dynamic_map_chain(themes, prompt: str):
    policy_groups = set()
    for theme in themes:
        if theme in THEMES_AND_POLICIES:
            policy_groups.update(THEMES_AND_POLICIES[theme])

    PolicyEnum = create_policy_enum(policy_groups)
    DynamicBriefSummary = create_brief_summary_model(PolicyEnum)

    SLLM = LLM.with_structured_output(DynamicBriefSummary, strict=True)

    prompt = (
        f"{prompt}\n\nAvailable Policies:\n\n- "
        + "\n- ".join(policy_groups)
        + "\n\nContext:\n\n{context}"
    )
    map_prompt = ChatPromptTemplate.from_messages([("system", prompt)])
    return map_prompt | SLLM


if __name__ == "__main__":
    test_document = """
    The Local Plan proposes a mass development north-west of Cambridge despite marked growth
    in the last twenty years or so following the previous New Settlement Study. In this period,
    the major settlement of Cambourne has been created - now over the projected 3,000 homes and
    Papworth Everard has grown beyond recognition. This in itself is a matter of concern.
    """
    test_themes = {"Homes", "Great Places"}

    dynamic_map_chain = create_dynamic_map_chain(test_themes, prompt=map_template)
    result = dynamic_map_chain.invoke({"context": test_document, "themes": test_themes})
    __import__("pprint").pprint(dict(result))
