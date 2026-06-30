from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from typing import List
import os

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Answer the question using ONLY
the context provided below. If the answer is not in the context, say
"I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer (cite sources as [Source: filename]):"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class RAGPipeline:
    def __init__(self, vectorstore: Chroma, retrieval_mode: str = "topk"):
        self.vectorstore = vectorstore
        self.retrieval_mode = retrieval_mode
        self._setup_llm()
        self.retriever = self._build_retriever()

    def _setup_llm(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  OPENAI_API_KEY not set. Generation will be skipped.")
            self.llm = None
            return
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def _build_retriever(self):
        if self.retrieval_mode == "mmr":
            return self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
            )
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

    def query(self, question: str) -> dict:
        # Always retrieve docs
        docs = self.retriever.invoke(question)
        sources = [
            {
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page", "N/A"),
                "chunk_id": d.metadata.get("chunk_id", "N/A"),
                "strategy": d.metadata.get("chunking_strategy", "N/A"),
                "preview": d.page_content[:100] + "...",
            }
            for d in docs
        ]

        # Generate answer if LLM available
        if self.llm is None:
            return {
                "question": question,
                "answer": "⚠️ No OPENAI_API_KEY set — retrieval only mode.",
                "retrieval_mode": self.retrieval_mode,
                "sources": sources,
                "num_sources": len(sources),
            }

        chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )
        answer = chain.invoke(question)

        return {
            "question": question,
            "answer": answer,
            "retrieval_mode": self.retrieval_mode,
            "sources": sources,
            "num_sources": len(sources),
        }

    def retrieve_only(self, question: str, k: int = 5) -> List[dict]:
        docs = self.vectorstore.similarity_search(question, k=k)
        return [
            {
                "chunk_id": d.metadata.get("chunk_id", ""),
                "source": d.metadata.get("source", ""),
                "preview": d.page_content[:150],
            }
            for d in docs
        ]
