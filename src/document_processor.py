from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from typing import List, Dict
import hashlib
import os
from datetime import datetime


class DocumentProcessor:
    def __init__(self):
        self.splitters = {
            "fixed": CharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50,
                separator="\n"
            ),
            "recursive": RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", " ", ""]
            ),
            "semantic": RecursiveCharacterTextSplitter(
                chunk_size=1024,
                chunk_overlap=200,
                separators=["\n## ", "\n### ", "\n\n", "\n", ". "]
            ),
        }

    def load_documents(self, data_dir: str) -> List[Document]:
        docs = []
        if not os.path.exists(data_dir):
            print(f"⚠️  Directory '{data_dir}' not found. Creating it...")
            os.makedirs(data_dir)
            return docs

        for filename in os.listdir(data_dir):
            filepath = os.path.join(data_dir, filename)
            try:
                if filename.endswith(".pdf"):
                    loader = PyPDFLoader(filepath)
                elif filename.endswith(".txt"):
                    loader = TextLoader(filepath, encoding="utf-8")
                else:
                    continue
                loaded = loader.load()
                docs.extend(loaded)
                print(f"✅ Loaded: {filename} ({len(loaded)} sections)")
            except Exception as e:
                print(f"❌ Failed: {filename} — {e}")

        print(f"\n📄 Total documents loaded: {len(docs)}")
        return docs

    def enrich_metadata(self, docs: List[Document], strategy: str) -> List[Document]:
        enriched = []
        for i, doc in enumerate(docs):
            doc.metadata.update({
                "chunk_id": hashlib.md5(doc.page_content.encode()).hexdigest()[:8],
                "chunk_index": i,
                "chunking_strategy": strategy,
                "timestamp": datetime.now().isoformat(),
                "char_count": len(doc.page_content),
                "word_count": len(doc.page_content.split()),
            })
            enriched.append(doc)
        return enriched

    def chunk(self, docs: List[Document], strategy: str = "recursive") -> List[Document]:
        if strategy not in self.splitters:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(self.splitters)}")
        splitter = self.splitters[strategy]
        chunks = splitter.split_documents(docs)
        chunks = self.enrich_metadata(chunks, strategy)
        sizes = [len(c.page_content) for c in chunks]
        print(f"\n🔪 Strategy : {strategy}")
        print(f"   Chunks   : {len(chunks)}")
        print(f"   Avg size : {sum(sizes) // len(sizes)} chars")
        return chunks

    def compare_strategies(self, docs: List[Document]) -> Dict:
        report = {}
        for strategy in self.splitters:
            chunks = self.chunk(docs, strategy)
            sizes = [len(c.page_content) for c in chunks]
            report[strategy] = {
                "total_chunks": len(chunks),
                "avg_size": sum(sizes) // len(sizes),
                "min_size": min(sizes),
                "max_size": max(sizes),
            }
        return report
