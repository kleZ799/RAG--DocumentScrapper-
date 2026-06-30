from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List
import time
import os


class VectorStoreManager:
    def __init__(self, use_openai: bool = False):
        if use_openai:
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            self.embedding_name = "openai-text-embedding-3-small"
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            self.embedding_name = "all-MiniLM-L6-v2"

        self.stores = {}
        print(f"🤖 Embedding model: {self.embedding_name}")

    def create_chroma(self, chunks: List[Document], collection_name: str) -> Chroma:
        persist_dir = f"./vectorstores/chroma_{collection_name}"
        os.makedirs(persist_dir, exist_ok=True)
        start = time.time()
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir
        )
        elapsed = time.time() - start
        print(f"\n✅ ChromaDB '{collection_name}' built in {elapsed:.2f}s")
        print(f"   Chunks indexed : {len(chunks)}")
        self.stores[collection_name] = vectorstore
        return vectorstore

    def load_chroma(self, collection_name: str) -> Chroma:
        persist_dir = f"./vectorstores/chroma_{collection_name}"
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir
        )
        self.stores[collection_name] = vectorstore
        return vectorstore

    def create_pinecone(self, chunks: List[Document], index_name: str = "rag-milestone1"):
        try:
            from pinecone import Pinecone, ServerlessSpec
            from langchain_pinecone import PineconeVectorStore
            api_key = os.environ.get("PINECONE_API_KEY")
            if not api_key:
                print("⚠️  PINECONE_API_KEY not set. Skipping.")
                return None
            pc = Pinecone(api_key=api_key)
            existing = [i.name for i in pc.list_indexes()]
            if index_name not in existing:
                pc.create_index(
                    name=index_name,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                time.sleep(10)
            vectorstore = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                index_name=index_name
            )
            print(f"✅ Pinecone '{index_name}' ready with {len(chunks)} chunks")
            return vectorstore
        except ImportError:
            print("⚠️  pinecone/langchain-pinecone not installed.")
            return None

    def benchmark(self, vectorstore, queries: List[str], k: int = 5) -> dict:
        latencies = []
        for query in queries:
            start = time.time()
            vectorstore.similarity_search(query, k=k)
            latencies.append(time.time() - start)
        return {
            "avg_latency_ms": round(sum(latencies) / len(latencies) * 1000, 2),
            "min_latency_ms": round(min(latencies) * 1000, 2),
            "max_latency_ms": round(max(latencies) * 1000, 2),
            "queries_tested": len(queries),
        }
