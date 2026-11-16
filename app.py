import os

from src.data_loader import load_all_documents
from src.search import RAGSearch
from src.vectorstore import FaissVectorStore
from src.embedding import EmbeddingPipeline


# Example usage
if __name__ == "__main__":
    # docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    # store.build_from_documents(docs)
    store.load()
    print(store.query("What is a load balancer?", top_k=3))

    rag_search = RAGSearch()
    query = "What is the difference between load balancer and gateway?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)