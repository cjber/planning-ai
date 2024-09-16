from langgraph.constants import Send

from planning_ai.chains.fix_chain import fix_chain
from planning_ai.chains.hallucination_chain import (
    HallucinationChecker,
    hallucination_chain,
)
from planning_ai.states import DocumentState, OverallState


def check_hallucination(state: DocumentState):
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
                "summary": state["summary"],
                "iteration": state["iteration"] + 1,
            }
        ]
    }


def map_hallucinations(state: OverallState):
    return [Send("check_hallucination", summary) for summary in state["summaries"]]


def fix_hallucination(state: DocumentState):
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
