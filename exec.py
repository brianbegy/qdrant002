

import sys
from dense import insert_dense, query_dense
from sparse import insert_sparse, query_sparse
from pandas import pandas as pd

action = sys.argv[1]
density = sys.argv[2]

print(f"Running {action} with {density} vectors.")

if action == "query":
    query_text = sys.argv[3]
    if query_text:
        print(f"Searching for: {query_text}")
        
        hits = query_sparse(query_text) if density == "sparse" else query_dense(query_text)
        print(hits)
        for hit in hits:
            print(hit.payload, "score:", hit.score)
            
elif action == "insert":
    comments = pd.read_csv('./data/comments.csv')["COMMENTS"].dropna().head(100)
    comment_length = len(comments)
    print(f"inserting {comment_length} comments using a {density} vector.")
    if density == "sparse":
        insert_sparse(comments)
    elif density == "dense":
        insert_dense(comments)
    else:
        print("Please enter a valid density type")  
else:
    print("Please enter a valid action")
