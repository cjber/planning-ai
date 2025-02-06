from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from planning_ai.common.utils import Paths
from planning_ai.llms.llm import O3Mini

with open(Paths.PROMPTS / "reduce.txt", "r") as f:
    reduce_template = f.read()

with open(Paths.PROMPTS / "reduce_final.txt", "r") as f:
    reduce_template_final = f.read()


reduce_prompt = ChatPromptTemplate([("system", reduce_template)])
reduce_prompt_final = ChatPromptTemplate([("system", reduce_template_final)])
reduce_chain = reduce_prompt | O3Mini | StrOutputParser()
reduce_chain_final = reduce_prompt_final | O3Mini | StrOutputParser()


if __name__ == "__main__":
    test_summary = """
        The author expresses concern over the proposed mass development north-west of Cambridge,
        highlighting the significant growth in the area over the past twenty years,
        particularly with the establishment of Cambourne and the expansion of Papworth Everard.
        """

    result = reduce_chain.invoke({"context": test_summary})

    print("Generated Report:")
    print(result)
