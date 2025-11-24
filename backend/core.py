from dotenv import load_dotenv
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langsmith import Client
from typing import List, Dict, Any
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever

load_dotenv()

INDEX_NAME="langchain-doc-index"
client = Client()

def run_llm(query: str, chat_history: List[Dict[str, Any]]):
    embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOllama(model="llama3.2:latest", verbose=True)

    retrieval_qa_chat_prompt = client.pull_prompt("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(llm=chat, prompt=retrieval_qa_chat_prompt)

    rephrase_prompt = client.pull_prompt("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"]
    }
    return new_result

if __name__ == "__main__":
    res = run_llm(query="")
    print(res["result"])