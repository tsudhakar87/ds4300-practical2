import pymongo
import numpy as np
from embedding_db_setup.embedder import Embedder

class MongoInstantiator(Embedder):
    def __init__(self):
        # Local MongoDB URI without Docker
        mongo_uri = "mongodb://diya:kada0201@localhost:27017"


        # Establish a connection to MongoDB
        self.client = pymongo.MongoClient(mongo_uri)
        
        # Specify the database and collection
        self.db_name = "notes_db"
        self.collection_name = "embeddings"
        
        # Get the collection from the database
        self.collection = self.client[self.db_name][self.collection_name]

        # Create an index on the 'text' field (for text search)
        self.create_index()

    def create_index(self):
        indexes = self.collection.index_information()
        if "text_text" not in indexes:
            self.collection.create_index([("text", pymongo.TEXT)])

    def clear_store(self):
        # Deletes all documents in the collection
        self.collection.delete_many({})

    def store_embedding(self, file_name: str, p_num: int, chunk: str, embedding: list):
        # Metadata for the document
        metadata = {
            "file_name": file_name,
            "page_num": p_num
        }

        # Document to store in the collection
        doc = {
            "text": chunk,
            "embedding": embedding,
            "metadata": metadata
        }

        # Insert the document into the collection
        self.collection.insert_one(doc)

    def search_embeddings(self, query: str, top_k: int):
        # Get the embedding for the query text
        query_embedding = np.array(self.get_embedding(query))

        # Fetch all documents from the collection
        all_docs = list(self.collection.find({}))

        scored_docs = []
        for doc in all_docs:
            doc_embedding = np.array(doc["embedding"])

            # Compute cosine similarity between query and document embeddings
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            scored_docs.append((similarity, doc))

        # Sort documents by similarity (descending order)
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Return the top K most similar documents
        return [doc for _, doc in scored_docs[:top_k]]
