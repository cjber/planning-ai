from langgraph.constants import Send
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from planning_ai.chains.map_chain import map_chain
from planning_ai.states import DocumentState, OverallState

anonymizer = AnonymizerEngine()
analyzer = AnalyzerEngine()


def remove_pii(document: str):
    results = analyzer.analyze(
        text=document,
        entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"],
        language="en",
    )
    document = anonymizer.anonymize(text=document, analyzer_results=results)
    return document


def generate_summary(state: DocumentState):
    state["document"] = remove_pii(state["document"])
    response = map_chain.invoke({"context": state["document"]})
    return {
        "summaries": [
            {
                "summary": response,
                "document": state["document"],
                "filename": state["filename"],
                "iteration": 1,
            }
        ]
    }


def map_summaries(state: OverallState):
    return [
        Send(
            "generate_summary",
            {"document": document, "filename": filename},
        )
        for document, filename in zip(state["documents"], state["filenames"])
    ]
