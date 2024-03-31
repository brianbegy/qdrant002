from get_client import get_client
from qdrant_client import models
from pandas import pandas as pd
from get_collection import get_collection
from get_encoder import get_encoder
import sys

action = sys.argv[1]

client = get_client()
collection_name = 'test_collection2'
encoder = get_encoder()
collection = get_collection(client, collection_name, encoder)

def query_dense(query_text):
    print("Searching for:", query_text)
    return client.search(
        collection_name=collection_name,
        query_vector=encoder.encode(query_text).tolist(),
        limit=10,
    )

def insert_dense():
    comments = pd.read_csv('./data/comments.csv')["COMMENTS"].dropna()
    print("inserting %d comments" % len(comments))
    client.upload_points(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=idx, vector=encoder.encode(doc).tolist(), payload={'idx':idx, 'comment':doc}
            )
            for idx, doc in enumerate(comments)
        ],
    )

if action == "query":
    if(len(sys.argv)>2 and len(sys.argv[2])>0):
        hits = query_dense(sys.argv[2])
        for hit in hits:
            print(hit.payload, "score:", hit.score)
    else: 
        print("Please enter a query")
elif action == "insert":
    insert_dense()
    print("done.")
