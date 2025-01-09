import json
import logging

from langchain_core.exceptions import OutputParserException
from langgraph.types import Send
from pydantic import BaseModel

from planning_ai.chains.fix_chain import fix_template
from planning_ai.chains.hallucination_chain import (
    HallucinationChecker,
    hallucination_chain,
)
from planning_ai.chains.map_chain import create_dynamic_map_chain
from planning_ai.states import DocumentState, OverallState

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BasicSummaryBroken(BaseModel):
    summary: str
    policies: None


ITERATIONS = 2


def check_hallucination(state: DocumentState):
    """Checks for hallucinations in the summary of a document.

    This function uses the `hallucination_chain` to evaluate the summary of a document.
    If the hallucination score is 1, it indicates no hallucination, and the summary is
    considered fixed. If the iteration count exceeds 5, the process is terminated.

    Args:
        state (DocumentState): The current state of the document, including its summary
            and iteration count.

    Returns:
        dict: A dictionary containing either a list of fixed summaries or hallucinations
        that need to be addressed.
    """
    logger.warning(f"Checking hallucinations for document {state['filename']}")
    # Stop trying after 2 iterations
    if state["iteration"] > ITERATIONS:
        state["iteration"] = 99
        state["hallucination"].score = 1
        return {"documents": [state]}

    try:
        response = hallucination_chain.invoke(
            {"document": state["document"], "summary": state["summary"].summary}
        )
    except (OutputParserException, json.JSONDecodeError) as e:
        logger.error(f"Failed to decode JSON: {e}.")
        state["iteration"] = 99
        state["hallucination"] = HallucinationChecker(score=1, explanation="INVALID")
        state["summary"] = BasicSummaryBroken(summary="INVALID", policies=None)
        return {"documents": [state]}
    if response.score == 1:
        return {"documents": [{**state, "hallucination": response}]}

    return {
        "documents": [
            {**state, "hallucination": response, "iteration": state["iteration"] + 1}
        ]
    }


def map_hallucinations(state: OverallState):
    """Maps summaries to the `check_hallucination` function.

    This function prepares a list of summaries to be checked for hallucinations by
    sending them to the `check_hallucination` function. Allows summaries to be checked
    in parrallel.

    Args:
        state (OverallState): The overall state containing all summaries.

    Returns:
        list: A list of Send objects directing each summary to the check_hallucination
        function.
    """
    return [Send("check_hallucination", document) for document in state["documents"]]


def fix_hallucination(state: DocumentState):
    """Attempts to fix hallucinations in a document's summary.

    This function uses the `fix_chain` to correct hallucinations identified in a summary.
    The corrected summary is then updated in the document state.

    Args:
        state (DocumentState): The current state of the document, including its summary
            and hallucination details.

    Returns:
        dict: A dictionary containing the updated summaries after attempting to fix
        hallucinations.
    """
    logger.warning(f"Fixing hallucinations for document {state['filename']}")
    fix_chain = create_dynamic_map_chain(state["themes"], fix_template)
    try:
        response = fix_chain.invoke(
            {
                "context": state["document"],
                "summary": state["summary"].summary,
                "explanation": state["hallucination"].explanation,
            }
        )
    except (OutputParserException, json.JSONDecodeError) as e:
        logger.error(f"Failed to decode JSON: {e}.")
        state["iteration"] = 99
        state["hallucination"] = HallucinationChecker(score=1, explanation="INVALID")
        state["summary"] = BasicSummaryBroken(summary="INVALID", policies=None)
        return {"documents": [state]}
    state["summary"] = response  # type: ignore
    return {"documents": [state]}


def map_fix_hallucinations(state: OverallState):
    """Maps hallucinations to the `fix_hallucination` function.

    This function filters out hallucinations that need fixing and prepares them to be
    sent to the `fix_hallucination` function. Allows hallucinations to be fixed in
    parrallel.

    Args:
        state (OverallState): The overall state containing all hallucinations.

    Returns:
        list: A list of Send objects directing each hallucination to the
        fix_hallucination function.
    """
    hallucinations = []
    if "documents" in state:
        hallucinations = [
            document
            for document in state["documents"]
            if document["hallucination"].score != 1
        ]
    return [
        Send("fix_hallucination", hallucination) for hallucination in hallucinations
    ]
