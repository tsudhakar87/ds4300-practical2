import ollama
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod

class Embedder(ABC):
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    @classmethod
    def change_embedding_model(cls, new_model):
        cls.embedding_model = new_model
        print(f"Embedding model changed to: {cls.embedding_model}")
  
    @classmethod
    def get_embedding(self, text: str) -> list:
        response = ollama.embeddings(model=self.embedding_model, prompt=text)
        return response["embedding"]
    
    @classmethod
    def indexing_speed():
        pass

    @abstractmethod
    def store_embedding(self, file: str, page: str, chunk: str, embedding: list):
        pass
    
    