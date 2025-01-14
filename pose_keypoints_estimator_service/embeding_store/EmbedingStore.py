import asyncio
import os
import sys
from PIL import Image
import uuid
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection,has_collection 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))



class EmbeddingStore:
    def __init__(self, collection_name='persons_keypoints', dim=399):
        # Connect to Milvus
        self.host = os.getenv('MILVUS_HOST', 'localhost')
        self.port = os.getenv('MILVUS_PORT', "19530")
        connections.connect("default", host=self.host, port=self.port)

        # Define schema and create collection if it doesn't exist
        self.collection_name = collection_name
        self.collection = self.create_or_get_collection(dim)

        self.create_index()

    def create_or_get_collection(self, dim):
        # Check if the collection exists
        if not has_collection(self.collection_name):
            field1 = FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=255)
            field2 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
            
            schema = CollectionSchema(fields=[field1, field2])
            collection = Collection(name=self.collection_name, schema=schema)
        else:
            collection = Collection(name=self.collection_name)
        return collection
    def create_index(self): 
        # Create an index on the 'embedding' field 
        index_params = { 
          "index_type": "IVF_FLAT",
          "metric_type": "L2",
          "params": {"nlist": 128} }
        self.collection.create_index(field_name="embedding", index_params=index_params)
    def save_to_milvus(self, embeddings):
        """Saves embeddings to Milvus and returns their keys."""
        ids = [f"{str(uuid.uuid4())}" for _ in range(len(embeddings))]
        data = [ { "id": ids[i], "embedding": embeddings[i] } for i , id in enumerate(ids) ]
        
        self.collection.insert(data)
        return ids

    def get_embedding_by_key(self, key):
        """Retrieves an embedding from Milvus using the key."""
        self.collection.load()
        expr = f"id == '{key}'"
        results = self.collection.query(expr, output_fields=["embedding"])
        return results

