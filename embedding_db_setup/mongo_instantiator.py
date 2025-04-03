import os
import pymongo 
import numpy as np
from embedding_db_setup.embedder import Embedder

class MongoInstantiator(Embedder):
    def __init__(self):
        mongo_uri = os.getenv("MONGO_URI")
        self.client = pymongo.MongoClient(mongo_uri)
        self.db_name = "notes_db"
        self.collection_name = "embeddings"
        self.collection = self.client[self.db_name][self.collection_name]

        self.create_index()

    def create_index(self):
        indexes = self.collection.index_information()
        if "text_text" not in indexes:
            self.collection.create_index([("text", pymongo.TEXT)])

    def clear_store(self):
        self.collection.delete_many({})

    def store_embedding(self, file_name: str, p_num: int, chunk: str, embedding: list):
        metadata = {
            "file_name": file_name,
            "page_num": p_num
        }

        doc = {
            "text": chunk,
            "embedding": embedding,
            "metadata": metadata
        }

        self.collection.insert_one(doc)

    def search_embeddings(self, query: str, top_k: int):
        query_embedding = np.array(self.get_embedding(query))

        all_docs = list(self.collection.find({}))

        scored_docs = []
        for doc in all_docs:
            doc_embedding = np.array(doc["embedding"])
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            scored_docs.append((similarity, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scored_docs[:top_k]]

