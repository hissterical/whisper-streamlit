from langchain_community.document_loaders import YoutubeLoader, DirectoryLoader, TextLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS, Chroma
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings
from typing import Any
import os
import shutil
import hashlib


class LangchainRAG:
    def __init__(self, embeddings: Any = None):
        self.embeddings = embeddings or OllamaEmbeddings(
            model="mxbai-embed-large")

    @staticmethod
    def get_OllamaEmbeddings(model: str = "mxbai-embed-large"):
        return OllamaEmbeddings(model=model)

    def get_related_docs_from_query(self, vectorStore: FAISS, query: str, num_docs=3):
        docs = vectorStore.similarity_search(query, num_docs)
        docs_page_content = "\n\n".join([doc.page_content for doc in docs])
        return docs_page_content


class FAISSVectorStore:
    def __init__(self, embeddings: Any = None):
        self.embeddings = embeddings or OllamaEmbeddings(
            model="mxbai-embed-large")

    def create_FAISS_from_texts(self, chunks: list[str]):
        vectorStore = FAISS.from_texts(chunks, self.embeddings)
        return vectorStore

    def create_FAISS_from_docs(self, docs: list[Document]):
        vectorStore = FAISS.from_documents(docs, self.embeddings)
        return vectorStore

    def save_vector_store(self, vectorStore: FAISS, path: str):
        vectorStore.save_local(path)

    def load_vector_store(self, path: str):
        return FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)


class ChromaVectorStore:
    def __init__(self, path: str, embeddings: Any = None):
        self.embeddings = embeddings or OllamaEmbeddings(
            model="mxbai-embed-large")
        self.path = path

    def create_chroma_db_from_docs(self, docs: list[Document]):
        db = Chroma.from_documents(
            docs,
            self.embeddings,
            persist_directory=self.path
        )
        db.persist()
        return db

    def create_chroma_db_from_texts(self, chunks: list[str]):
        db = Chroma.from_texts(
            chunks,
            self.embeddings,
            persist_directory=self.path
        )
        db.persist()
        return db

    def clear_chroma_db(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    @staticmethod
    def add_docs_to_chroma(db: Chroma, docs: list[Document]):
        # Calculate Page IDs.
        chunks_with_ids = ChromaVectorStore.calculate_chunk_ids(docs)

        # Add or Update the documents.
        # IDs are always included by default
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        existing_content_hashes = set(
            existing_items["metadatas"].get("content_hash", None))
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Only add documents that don't exist in the DB.
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids and chunk.metadata["content_hash"] not in existing_content_hashes:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            db.persist()
        else:
            print("✅ No new documents to add")

    @staticmethod
    def calculate_chunk_ids(chunks):

        # This will create IDs like "data/monopoly.pdf:6:2"
        # Page Source : Page Number : Chunk Index

        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            # If the page ID is the same as the last one, increment the index.
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # Calculate the chunk ID.
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            # Add it to the page meta-data.
            chunk.metadata["id"] = chunk_id

            # hash page content and add to metadata as a way of ensuring content isn't changed, and if so, update the document
            chunk.metadata["content_hash"] = hashlib.md5(
                chunk.page_content.encode()).hexdigest()

        return chunks


class DocumentLoaders:
    def __init__(self, splitter: CharacterTextSplitter | None = None):
        self.splitter = splitter or CharacterTextSplitter(
            chunk_size=500, chunk_overlap=100)

    def split_docs(self, docs: list[Document]):
        if len(docs) == 1:
            return self.splitter.split_text(docs[0].page_content)
        else:
            return self.splitter.split_documents(docs)

    def set_splitter(self, splitter: CharacterTextSplitter):
        self.splitter = splitter
        return self

    def get_chunks_from_video(self, video_url: str):
        loader = YoutubeLoader.from_youtube_url(
            video_url, add_video_info=False)
        transcript = loader.load()  # returns list of documents
        chunks = self.split_docs(transcript)
        return chunks

    def get_docs_from_pdf(self, pdf_path: str):
        loader = PyPDFDirectoryLoader(pdf_path)
        docs = loader.load()
        chunks = self.split_docs(docs)
        return chunks

    def get_docs_from_directory(self, directory_path: str, glob: str = "*.txt"):
        loader = DirectoryLoader(directory_path, glob=glob)
        documents = loader.load()
        return self.split_docs(documents)

    def get_docs_from_file(self, file_path: str):
        loader = TextLoader(file_path)
        documents = loader.load()
        return self.split_docs(documents)

    def get_docs_from_url(self, url: str):
        loader = WebBaseLoader(url)
        documents = loader.load()
        return self.split_docs(documents)
