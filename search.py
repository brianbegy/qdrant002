from get_client import get_client
from get_encoder import get_encoder
from get_collection import get_collection


client = get_client()
collection_name = 'test_collection'
encoder = get_encoder()
collection = get_collection(client, collection_name, encoder)

query = "salary or pay"

hits = client.search(
    collection_name=collection_name,
    query_vector=encoder.encode(query).tolist(),
    limit=30,
)

for hit in hits:
    print(hit.payload, "score:", hit.score)