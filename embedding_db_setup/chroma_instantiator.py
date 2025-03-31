from .embedder import Embedder
import chromadb
from chromadb.utils import embedding_functions

class ChromaInstantiator(Embedder):
    def __init__(self):
        """Initialize Chroma client and collection."""
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("embedding_collection")

    def change_embedding_model(self, new_model: str):
        """Change the embedding model."""
        self.embedding_model = new_model
        print(f"Embedding model changed to: {self.embedding_model}")

    def create_index(self):
        """Create an index in Chroma (if needed)."""
        # Index creation logic (if needed) can go here
        print("Index created.")

    def store_embedding(self, file: str, page: str, chunk: str, embedding: list):
        """Store the embedding in Chroma."""
        doc_id = f"{file}_{page}_{hash(chunk)}"
        
        # Ensure embedding is not None before attempting to store it
        if embedding is None:
            print(f"Warning: Embedding for chunk '{chunk}' is None. Skipping this chunk.")
            return

        try:
            # Add the embedding to Chroma collection
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],  # List of embeddings
                metadatas=[{"file": file, "page": page, "chunk": chunk}]  # Metadata for each embedding
            )
            print(f"Stored embedding for chunk {chunk} from file {file} at page {page}.")
        except Exception as e:
            print(f"Error storing embedding: {e}")

    def get_embedding(self, text: str) -> list:
        """Get the embedding for a given text."""
        try:
            # Generate the embedding using the parent class method (which handles Ollama)
            embedding = super().get_embedding(text)
            if embedding is None:
                print(f"Error: Embedding for text is None.")
                return None
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
