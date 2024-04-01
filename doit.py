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
print(encoder)
collection = get_collection(client, collection_name, encoder)

if action == "query":
    if(len(sys.argv)>2 and len(sys.argv[2])>0):
        query = sys.argv[2]
        print("Searching for:", query)
        hits = client.search(
            collection_name=collection_name,
            query_vector=encoder.encode(query).tolist(),
            limit=10,
        )

        for hit in hits:
            print(hit.payload, "score:", hit.score)
    else: 
        print("Please enter a query")
elif action == "insert":
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
    print("done.")
