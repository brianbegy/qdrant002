from qdrant_client import models
from get_encoder import get_encoder

def get_collection(client, collection_name, encoder=get_encoder()):
    try:
        collection = client.get_collection(collection_name)
        return collection
    except:
        return client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "comment_dense": models.VectorParams(
                size=encoder.get_sentence_embedding_dimension(),
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