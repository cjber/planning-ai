from langgraph.types import Send

from planning_ai.chains.fix_chain import fix_template
from planning_ai.chains.hallucination_chain import hallucination_chain
from planning_ai.chains.map_chain import create_dynamic_map_chain
from planning_ai.logging import logger
from planning_ai.states import DocumentState, OverallState

MAX_ATTEMPTS = 3


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
    logger.info(f"Checking hallucinations for document {state['filename']}")

    if state["processed"] or (state["refinement_attempts"] >= MAX_ATTEMPTS):
        logger.error(f"Max attempts exceeded for document: {state['filename']}")
        return {"documents": [{**state, "failed": True, "processed": True}]}
    elif not state["is_hallucinated"]:
        logger.info(f"Finished processing document: {state['filename']}")
        return {"documents": [{**state, "processed": True}]}

    try:
        response = hallucination_chain.invoke(
            {"document": state["document"], "summary": state["summary"].summary}
        )
        is_hallucinated = response.score == 0
        refinement_attempts = state["refinement_attempts"] + 1
    except Exception as e:
        logger.error(f"Failed to decode JSON {state['filename']}: {e}")
        return {
            "documents": [
                {
                    **state,
                    "summary": "",
                    "refinement_attempts": 0,
                    "is_hallucinated": True,
                    "failed": True,
                    "processed": True,
                }
            ]
        }

    out = {
        **state,
        "hallucination": response,
        "refinement_attempts": refinement_attempts,
        "is_hallucinated": is_hallucinated,
    }
    logger.info(f"Hallucination for {state['filename']}: {is_hallucinated}")
    return (
        {"documents": [{**out, "processed": False}]}
        if is_hallucinated
        else {"documents": [{**out, "processed": True}]}
    )


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
    except Exception as e:
        logger.error(f"Failed to decode JSON {state['filename']}: {e}.")
        return {
            "documents": [
                {
                    **state,
                    "summary": "",
                    "refinement_attempts": 0,
                    "is_hallucinated": True,
                    "failed": True,
                    "processed": True,
                }
            ]
        }
    return {"documents": [{**state, "summary": response}]}


def map_check(state: OverallState):
    return [Send("check_hallucination", doc) for doc in state["documents"]]


def map_fix(state: OverallState):
    return [
        Send("fix_hallucination", doc)
        for doc in state["documents"]
        if doc["is_hallucinated"] and not doc["processed"]
    ]
