from embedder import Embedder
import chromadb
from chromadb.utils import embedding_functions



class ChromaInstantiator(Embedder):
    def __init__(self, path="./chroma_db", collection_name="course_notes"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def clear_chroma_store(self):
        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.get_or_create_collection(name=self.collection.name)
        print("ChromaDB store cleared.")

    def store_embedding(self, file: str, page: str, chunk: str, embedding: list):

        doc_id = f"{file}_{page}_{hash(chunk)}"
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding], 
            metadatas=[{"file": file, "page": page, "chunk": chunk}]
        )

    def query_chroma(self, query_text: str, top_k: int = 5):
        embedding = self.get_embedding(query_text)  
        results = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        return results["metadatas"]

        