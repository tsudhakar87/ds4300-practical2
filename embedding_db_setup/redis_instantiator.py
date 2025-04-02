import redis
import numpy as np
from redis.commands.search.query import Query
from embedding_db_setup.embedder import Embedder

# embedding constants
VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"
class RedisInstantiator(Embedder):
    def __init__(self):
        super().__init__()
        self.client = redis.Redis(host="localhost", port=6379, db=0)
        
    # change embedding model
    def change_embedding_model(self, new_model):
        self.embedding_model = new_model
        
    # used to clear the redis vector store
    def clear_redis_store(self):
        print("Clearing existing Redis store...")
        self.client.flushdb()
        print("Redis store cleared.")

    # create an HNSW index in Redis
    def create_hnsw_index(self):
        try:
            self.client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
        except redis.exceptions.ResponseError:
            pass

        self.client.execute_command(
            f"""
            FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
            SCHEMA text TEXT
            embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
            """
        )
        print("Index created successfully.")

    # store the embedding in Redis
    def store_embedding(self, file: str, page: str, chunk: str, embedding: list):
        key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
        self.client.hset(
            key,
            mapping={
                "file": file,
                "page": page,
                "chunk": chunk,
                "embedding": np.array(
                    embedding, dtype=np.float32
                ).tobytes(),  # Store as byte array
            },
        )
        print(f"Stored embedding for: {chunk}")
        
    # query redis db
    def query_redis(self, query_text: str):
        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("id", "vector_distance")
            .dialect(2)
        )
        query_text = "Efficient search in vector databases"
        embedding = self.get_embedding(query_text)
        res = self.client.ft(INDEX_NAME).search(
            q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
        )

        for doc in res.docs:
            print(f"{doc.id} \n ----> {doc.vector_distance}\n")
    