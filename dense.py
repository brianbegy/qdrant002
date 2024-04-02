from get_client import get_client
from qdrant_client import models
from pandas import pandas as pd
import sys
from sentence_transformers import SentenceTransformer
from get_collection import get_collection
from get_encoder import get_encoder

action = sys.argv[1]

client = get_client()
encoder = get_encoder()

def query_dense(query_text, collection_name):
    # get_dense_collection(client, collection_name)

    # Create the query vector
    query_vector = models.NamedVector(name= 'comment_dense', vector=encoder.encode(query_text).tolist())
    # Perform the search
    results = client.search(collection_name=collection_name, query_vector=query_vector, limit=100)

    return list(map(lambda x: {'raw': x.payload['raw'], 'score': x.score}, results))


def insert_dense(comments, collection_name):
    get_collection(client, collection_name)
    points = []
    
    for idx, doc in enumerate(comments):
        points.append(models.PointStruct(
            id=idx,
            vector={'comment_dense': encoder.encode(doc).tolist()},
            payload={'idx': idx, 'raw': doc}
        ))  
        
    client.upload_points(
        collection_name=collection_name,
        points=points,
        wait=True
    )
