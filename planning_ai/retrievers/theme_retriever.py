import logging
from pathlib import Path

from chromadb import PersistentClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

from planning_ai.llms.llm import LLM

# See: https://consultations.greatercambridgeplanning.org/greater-cambridge-local-plan-preferred-options/supporting-documents

PDFS = {
    "Biodiversity and Green Spaces": "https://consultations.greatercambridgeplanning.org/sites/gcp/files/2021-11/TPBiodiversityandGreenSpacesAug21v2Nov21_0.pdf",
    "Climate Change": "https://consultations.greatercambridgeplanning.org/sites/gcp/files/2021-11/TPClimateChangeAug21v2Nov21_0.pdf",
    "Great Places": "https://consultations.greatercambridgeplanning.org/sites/gcp/files/2021-11/TPGreatPlacesAug21v1Aug21.pdf",
    "Homes": "https://consultations.greatercambridgeplanning.org/sites/gcp/files/2021-11/TPHomesAug21v2Nov21.pdf",
    "Infrastructure": "https://consultations.greatercambridgeplanning.org/sites/gcp/files/2021-11/TPInfrastructureAug21v2Nov21.pdf",
    "Jobs": "https://consultations.greatercambridgeplanning.org/sites/gcp/files/2021-11/TPJobsAug21v2Nov21.pdf",
    # "Strategy topic paper": "https://consultations.greatercambridgeplanning.org/sites/gcp/files/2021-11/TPStrategyAug21v3Nov21_0.pdf",
    "Wellbeing and Social Inclusion": "https://consultations.greatercambridgeplanning.org/sites/gcp/files/2021-11/TPWellbeingAug21v2Nov21.pdf",
}


class Grade(BaseModel):
    """Binary score for relevance check."""

    binary_score: str = Field(description="Relevance score 'yes' or 'no'")


def create_db():
    chroma_dir = Path("./chroma_themesdb")
    if chroma_dir.exists():
        persistent_client = PersistentClient(path="./chroma_themesdb")
        vectorstore = Chroma(
            client=persistent_client,
            collection_name="themes-chroma",
            embedding_function=OpenAIEmbeddings(),
        )

    else:
        docs = []
        for name, pdf in PDFS.items():
            doc = PyPDFLoader(pdf).load()[5:]
            for d in doc:
                d.metadata["theme"] = name
            docs.extend(doc)

        logging.warning(f"Building ChromaDB...")
        vectorstore = Chroma.from_documents(
            documents=docs,
            collection_name="themes-chroma",
            embedding=OpenAIEmbeddings(),
            persist_directory="./chroma_themesdb",
        )
    return vectorstore


grade_template = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the original document: {document} \n
        If the retrieved document contains keyword(s) or semantic meaning related to the original, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the retrieved document is relevant to the original.""",
    input_variables=["context", "document"],
)


SLLM = LLM.with_structured_output(Grade, strict=True)
grade_chain = grade_template | SLLM

vectorstore = create_db()
theme_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
logging.warning(f"Finished building ChromaDB...")

if __name__ == "__main__":
    test_content = """
    We would certainly support this and would emphasise the importance of trying
    to solve the severance problems created by the M11 and A14.
    """

    len(theme_retriever.invoke(input=test_content))
