from sentence_transformers import SentenceTransformer
 
def get_encoder():
#    return SentenceTransformer('all-MiniLM-L6-v2') #384 dimension space
    return SentenceTransformer('all-mpnet-base-v2') #384 dimension space
