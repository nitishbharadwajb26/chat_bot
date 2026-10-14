import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_fixed(text, chunk_size, overlap_size: int = 0):
    """
    Splits the input text into chunks of a fixed size.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The size of each chunk.

    Returns:
        list: A list of text chunks.
    """
    print(f"Chunking text into fixed size chunks of {chunk_size} characters.")
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def chunk_by_sentences(text, chunk_size, overlap_size: int = 0):
    """
    Splits the input text into chunks based on sentences, ensuring that each chunk does not exceed the specified maximum size.
    """

    # Split the text into sentences using regex
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def chunk_overlapping(text, chunk_size, overlap_size):
    """
    Splits the input text into overlapping chunks.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The size of each chunk.
        overlap_size (int): The number of characters to overlap between chunks.

    Returns:
        list: A list of overlapping text chunks.
    """

    print(f"Chunking text into overlapping chunks of {chunk_size} characters with an overlap of {overlap_size} characters.")
    if overlap_size >= chunk_size:
        raise ValueError("Overlap size must be smaller than chunk size.")

    step = chunk_size - overlap_size
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]

def chunk_recursive(text, chunk_size, overlap_size):
    """
    Splits the input text into chunks using a recursive character text splitter.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The size of each chunk.
        overlap_size (int): The number of characters to overlap between chunks.

    Returns:
        list: A list of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
        separators=["\n\n", "\n", " ", "", "."]
    )
    return splitter.split_text(text)

STRATEGIES = {
    "fixed": chunk_fixed,
    "overlap": chunk_overlapping,
    "sentences": chunk_by_sentences,
    "recursive": chunk_recursive,
}

def chunk_text(text, strategy="fixed", **kwargs):
    """
    Chunks the input text based on the specified strategy.

    Args:
        text (str): The input text to be chunked.
        strategy (str): The chunking strategy to use. Options are "fixed", "overlap", "sentences", and "recursive".
        chunk_size (int): The size of each chunk.
        overlap_size (int): The number of characters to overlap between chunks (only applicable for "overlap" and "recursive" strategies).

    Returns:
        list: A list of text chunks.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")

    print(f"Chunking text using strategy: {strategy}")

    chunk_text = STRATEGIES[strategy](text, **kwargs)
    return [c.strip() for c in chunk_text if c.strip()]  # Remove empty chunks
    