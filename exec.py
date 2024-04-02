

import sys
from dense import insert_dense, query_dense
from sparse import insert_sparse, query_sparse
from pandas import pandas as pd

action = sys.argv[1]
density = sys.argv[2]

collection_name = "the_collection_2"
print(f"Running {action} with {density} vectors.")

if action == "query":
    query_text = sys.argv[3]
    if query_text:
        print(f"Searching for: {query_text}")
        
        hits = query_sparse(query_text, collection_name) if density == "sparse" else query_dense(query_text, collection_name)
 
        print(f"{len(hits)} hits")
        
        for hit in hits:
            print('score:', hit['score'], 'raw:', hit['raw'])
            
elif action == "insert":
    comments = pd.read_csv('./data/comments.csv')["COMMENTS"].dropna().head(1000)
    comment_length = len(comments)
    print(f"inserting {comment_length} comments using a {density} vector.")
    if density == "sparse":
        insert_sparse(comments, collection_name)
    elif density == "dense":
        insert_dense(comments, collection_name)
    else:
        print("Please enter a valid density type")  
else:
    print("Please enter a valid action")
