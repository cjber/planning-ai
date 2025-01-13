from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from planning_ai.common.utils import Paths
from planning_ai.llms.llm import LLM

with open(Paths.PROMPTS / "reduce.txt", "r") as f:
    reduce_template = f.read()


reduce_prompt = ChatPromptTemplate([("system", reduce_template)])
reduce_chain = reduce_prompt | LLM | StrOutputParser()


if __name__ == "__main__":
    test_summary = """
        Summary:

        The author expresses concern over the proposed mass development north-west of Cambridge,
        highlighting the significant growth in the area over the past twenty years,
        particularly with the establishment of Cambourne and the expansion of Papworth Everard.
        """

    result = reduce_chain.invoke({"context": test_summary})

    print("Generated Report:")
    print(result)
