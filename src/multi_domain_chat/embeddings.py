from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(MODEL_NAME)

def embed_texts(texts):
    """
    Embeds the input text using the SentenceTransformer model.

    Args:
        text (str): The input text to be embedded.

    Returns:
        list: A list of embeddings for the input text.
    """
    return _model.encode(texts, batch_size=32, normalize_embeddings=True).tolist()


def embed_query(query):
    """
    Embeds the input query using the SentenceTransformer model.

    Args:
        query (str): The input query to be embedded.

    Returns:
        list: A list of embeddings for the input query.
    """
    return embed_texts([query])[0]