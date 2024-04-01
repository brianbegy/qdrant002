from get_client import get_client
from qdrant_client import models
from pandas import pandas as pd
from get_collection import get_collection
from get_encoder import get_encoder
import sys

action = sys.argv[1]

client = get_client()
encoder = get_encoder()

def query_dense(query_text, collection_name="dense_collection"):
    collection = get_collection(client, collection_name, encoder)
    return client.search(
        collection_name=collection_name,
        query_vector=encoder.encode(query_text).tolist(),
        limit=10,
    )

def insert_dense(comments, collection_name = "dense_collection"):
    collection = get_collection(client, collection_name, encoder)
    client.upload_points(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=idx, vector=encoder.encode(doc).tolist(), payload={'idx':idx, 'comment':doc}
            )
            for idx, doc in enumerate(comments)
        ],
    )
