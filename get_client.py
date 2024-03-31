from qdrant_client import QdrantClient


def get_client():
    return QdrantClient("localhost", port=6333)
