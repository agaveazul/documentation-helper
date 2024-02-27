import os
from dotenv import load_dotenv

load_dotenv()
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from typing import Any, Dict, List




def run_llm(query: str) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=os.environ['INDEX_NAME'], embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    return qa({"query": query})

if __name__ == "__main__":
    print(run_llm(query="What is RetrievalQA chain?"))
