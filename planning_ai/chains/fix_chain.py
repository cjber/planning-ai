from planning_ai.chains.map_chain import create_dynamic_map_chain
from planning_ai.common.utils import Paths

with open(Paths.PROMPTS / "fix_hallucination.txt", "r") as f:
    fix_template = f.read()

if __name__ == "__main__":
    test_document = """
    The Local Plan proposes a mass development north-west of Cambridge despite marked growth
    in the last twenty years or so following the previous New Settlement Study. In this period,
    the major settlement of Cambourne has been created - now over the projected 3,000 homes and
    Papworth Everard has grown beyond recognition. This in itself is a matter of concern.
    """
    test_themes = {"Great Places", "Homes", "Climate Change"}
    fix_chain = create_dynamic_map_chain(test_themes, fix_template)
    result = fix_chain.invoke(
        {
            "summary": "This plan is great because they are building a nuclear power plant.",
            "explanation": "The original response does not mention a nuclear power plant, and appears to view the plan negatively",
            "context": test_document,
        }
    )
    __import__("pprint").pprint(dict(result))
