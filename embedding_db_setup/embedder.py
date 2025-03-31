import time
import ollama
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod

class Embedder(ABC):
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # default
    llm_model = "mistral:7b" # default

    @classmethod
    def change_embedding_model(cls, new_model: str):
        cls.embedding_model = SentenceTransformer(new_model)
        print(f"Embedding model changed to: {cls.embedding_model}")
        
    @classmethod
    def change_llm_model(cls, new_model: str):
        cls.llm_model = new_model
        print(f"LLM model changed to: {cls.llm_model}")

    @classmethod
    def get_embedding(cls, text: str) -> list:
        try:
            start_time = time.time()
            response = cls.embedding_model.encode(text).tolist()  # Use SentenceTransformer's encode method
            end_time = time.time()
            print(f"Embedding generated in {end_time - start_time:.4f} seconds")

            return response
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    @classmethod
    def chat_with_model(cls, prompt):
        response = ollama.chat(model=cls.llm_model, messages=[{"role": "user", "content": prompt}])
        print(f"Model: {cls.llm_model}\nResponse: {response['message']['content']}")
    
    @classmethod
    def indexing_speed(cls, store_function, *args):
        start_time = time.time()
        store_function(*args)
        end_time = time.time()
        print(f"Indexing completed in {end_time - start_time:.4f} seconds")

    @abstractmethod
    def store_embedding(self, file: str, page: str, chunk: str, embedding: list):
        pass