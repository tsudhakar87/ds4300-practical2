import ollama
from abc import ABC, abstractmethod

class Embedder(ABC):
    embedding_model = "nomic-embed-text"

    @classmethod
    def change_embedding_model(cls, new_model):
        cls.embedding_model = new_model
        print(f"Embedding model changed to: {cls.embedding_model}")
  
    @classmethod
    def get_embedding(self, text: str) -> list:
        response = ollama.embeddings(model=self.embedding_model, prompt=text)
        return response["embedding"]

    @abstractmethod
    def store_embedding(self, file: str, page: str, chunk: str, embedding: list):
        pass