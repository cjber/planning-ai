from langgraph.constants import Send

from planning_ai.chains.fix_chain import fix_chain
from planning_ai.chains.hallucination_chain import (
    HallucinationChecker,
    hallucination_chain,
)
from planning_ai.states import DocumentState, OverallState


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
    if state["iteration"] > 5:
        state["iteration"] = -99
        return {"summaries_fixed": [state]}

    response: HallucinationChecker = hallucination_chain.invoke(
        {"document": state["document"], "summary": state["summary"]}
    )  # type: ignore
    if response.score == 1:
        return {"summaries_fixed": [state]}

    return {
        "hallucinations": [
            {
                "hallucination": response,
                "document": state["document"],
                "filename": state["filename"],
                "summary": state["summary"],
                "iteration": state["iteration"] + 1,
            }
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
    return [Send("check_hallucination", summary) for summary in state["summaries"]]


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
    response = fix_chain.invoke(
        {
            "context": state["document"],
            "summary": state["summary"],
            "explanation": state["hallucination"],
        }
    )
    state["summary"] = response  # type: ignore
    return {
        "summaries": [
            {
                "document": state["document"],
                "filename": state["filename"],
                "summary": state["summary"],
                "iteration": state["iteration"],
            }
        ]
    }


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
    if "hallucinations" in state:
        hallucinations = [
            hallucination
            for hallucination in state["hallucinations"]
            if hallucination["hallucination"].score != 1
        ]
    return [
        Send("fix_hallucination", hallucination) for hallucination in hallucinations
    ]
