from get_client import get_client
from qdrant_client import models
from pandas import pandas as pd
from get_collection import get_collection
from get_encoder import get_encoder
import sys

action = sys.argv[1]

client = get_client()
encoder = get_encoder()

def query_dense(query_text, collection_name):
    get_collection(client, collection_name, encoder)
    query_dense_vector = encoder.encode(query_text).tolist()
    return client.search_batch(
        collection_name=collection_name,
        requests=[
            models.SearchRequest(
                vector=models.NamedVector(
                    name="comment_dense",
                    vector=query_dense_vector,
                ),
                limit=10,
            ),
        ])

def insert_dense(comments, collection_name):
    get_collection(client, collection_name, encoder)
    print(collection_name)
    client.upload_points(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=idx, vector=encoder.encode(doc).tolist(), payload={'idx':idx, 'comment_dense':doc}
            )
            for idx, doc in enumerate(comments)
        ],
    )
