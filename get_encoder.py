from sentence_transformers import SentenceTransformer
 
def get_encoder():
    return SentenceTransformer('all-MiniLM-L6-v2')
