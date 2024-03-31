from qdrant_client import models

def get_collection(client, collection_name, encoder):
    try:
      return client.get_collection(collection_name=collection_name)
    except Exception as e:
        collection = client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=encoder.get_sentence_embedding_dimension(), distance=models.Distance.COSINE),
        )
        return collection