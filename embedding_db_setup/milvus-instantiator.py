from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from embedder import Embedder

class MilvusInstantiator(Embedder):
    def __init__(self, collection_name="class-materials"):
        super().__init__()
        self.collection_name = collection_name

        # Connect to Millie
        connections.connect("default", port=6380, db=0)

        # Define collection schema (if not already created)
        field = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="file", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="page", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)  # Matches Ollama model output
        ]
        schema = CollectionSchema(field, description="DS4300 course notes embeddings")

        # Create collection
        self.collection = Collection(name=self.collection_name, schema=schema)
        self.collection.load()

    def store_embedding(self, file: str, page: str, chunk: str, embedding: list):
        """Store an embedding in Milvus."""
        entity = {
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding": embedding
        }
        self.collection.insert(entity)
        print(f"Stored embedding for file: {file}, page: {page}")

    def search(self, query: str, top_k=2):
        """Search for similar documents using Milvus."""
        query_embedding = self.get_embedding(query)
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_field=["file", "page", "chunk"]
        )
        return results