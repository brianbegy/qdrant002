import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from get_client import get_client
from qdrant_client import models

model_id = "naver/splade-cocondenser-ensembledistil"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id)

def compute_vector(text):
    """
    Computes a vector from logits and attention mask using ReLU, log, and max operations.
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    output = model(**tokens)
    logits, attention_mask = output.logits, tokens.attention_mask
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)
    max_val, _ = torch.max(weighted_log, dim=1)
    vec = max_val.squeeze()

    return vec, tokens

def extract_and_map_sparse_vector(vector, tokenizer):
    """
    Extracts non-zero elements from a given vector and maps these elements to their human-readable tokens using a tokenizer. The function creates and returns a sorted dictionary where keys are the tokens corresponding to non-zero elements in the vector, and values are the weights of these elements, sorted in descending order of weights.

    This function is useful in NLP tasks where you need to understand the significance of different tokens based on a model's output vector. It first identifies non-zero values in the vector, maps them to tokens, and sorts them by weight for better interpretability.

    Args:
    vector (torch.Tensor): A PyTorch tensor from which to extract non-zero elements.
    tokenizer: The tokenizer used for tokenization in the model, providing the mapping from tokens to indices.

    Returns:
    dict: A sorted dictionary mapping human-readable tokens to their corresponding non-zero weights.
    """

    # Extract indices and values of non-zero elements in the vector
    cols = vector.nonzero().squeeze().cpu().tolist()
    weights = vector[cols].cpu().tolist()

    # Map indices to tokens and create a dictionary
    idx2token = {idx: token for token, idx in tokenizer.get_vocab().items()}
    token_weight_dict = {
        idx2token[idx]: round(weight, 2) for idx, weight in zip(cols, weights)
    }

    # Sort the dictionary by weights in descending order
    sorted_token_weight_dict = {
        k: v
        for k, v in sorted(
            token_weight_dict.items(), key=lambda item: item[1], reverse=True
        )
    }

    return sorted_token_weight_dict

client = get_client()

def get_sparse_collection(client, collection_name):
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config={},
        sparse_vectors_config={
            "comment": models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False,
                )
            )
        },
    )


def query_sparse(query_text, collection_name="sparse_collection"):
    get_sparse_collection(client, collection_name)
    print("Searching for:", query_text)

    query_vec, query_tokens = compute_vector(query_text)
    query_vec.shape

    print(extract_and_map_sparse_vector(query_vec, tokenizer))
    query_indices = query_vec.nonzero().numpy().flatten().tolist()
    query_values = query_vec.detach().numpy()[query_indices].tolist() 
    
    return client.search(collection_name=collection_name,
        query_vector=models.NamedSparseVector(
            name="comment",
            vector=models.SparseVector(
                indices=query_indices,
                values=query_values,
            ),
        ),
        with_vectors=True,
    )
        
def insert_sparse(comments, collection_name="sparse_collection"):
    get_sparse_collection(client, collection_name)
    for idx, doc in enumerate(comments):
        print(f"Inserting comment {idx}")
        vec, tokens = compute_vector(doc)
        
        print(extract_and_map_sparse_vector(vec, tokenizer))
        indices = vec.nonzero().numpy().flatten()
        values = vec.detach().numpy()[indices]
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=idx,
                    payload={
                        'index':idx, 'raw':doc},
                    vector={
                        "comment": models.SparseVector(
                            indices=indices.tolist(), values=values.tolist()
                        )
                    },
                )
            ],
        )