from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from multi_domain_chat.parsers import parse_file
from multi_domain_chat.chunking import chunk_text
from multi_domain_chat.embeddings import embed_texts

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    data = await file.read()
    file_path = UPLOAD_DIR / file.filename
    file_path.write_bytes(data)

    try:
        parsed_content = parse_file(file_path)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if not parsed_content.strip():
        raise HTTPException(400, "Uploaded file is empty or could not be parsed.")

    return {
        "source": file.filename,
        "characters": len(parsed_content),
        "preview": parsed_content[:500],
    }

@app.get("/chunk_preview")
async def chunk_preview(file_name: str, strategy:str = "overlap", chunk_size: int = 1000, overlap_size: int = 200):
    file_path = UPLOAD_DIR / file_name
    if not file_path.exists():
        raise HTTPException(404, "File not found.")

    try:
        parsed_content = parse_file(file_path)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if not parsed_content.strip():
        raise HTTPException(400, "Uploaded file is empty or could not be parsed.")

    chunks = chunk_text(parsed_content, strategy, chunk_size=chunk_size, overlap_size=overlap_size)

    return {
        "source": file_name,
        "total_chunks": len(chunks),
        "avg_length": sum(len(c) for c in chunks) // max(len(chunks), 1),
        "chunks": chunks[:5],  # Return only the first 5 chunks for preview
    }

@app.get("embed_preview")
async def embed_preview(file_name: str):
    file_path = UPLOAD_DIR / file_name
    if not file_path.exists():
        raise HTTPException(404, "File not found.")

    try:
        parsed_content = parse_file(file_path)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if not parsed_content.strip():
        raise HTTPException(400, "Uploaded file is empty or could not be parsed.")


    chunks = chunk_text(parsed_content, "recursive", chunk_size=1000, overlap_size=200)
    preview_chunks = chunks[:5]

    embeddings = embed_texts(preview_chunks)

    return {
        "source": file_name,
        "total_chunks": len(chunks),
         "dimension": len(embeddings[0]),
        "first_vector_sample": embeddings[0][:8],
    }