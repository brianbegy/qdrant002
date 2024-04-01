from qdrant_client import models

def get_collection(client, collection_name, encoder):
    return client.recreate_collection(
    collection_name=collection_name,
    vectors_config={
        "comment_dense": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
        )
    },
    sparse_vectors_config={
        "comment_sparse": models.SparseVectorParams(
            index=models.SparseIndexParams(
                on_disk=False,
            )
        )
    },
)