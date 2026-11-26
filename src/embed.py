r"""
Day 3 - Stage 2 of the pipeline: chunks -> vectors -> ChromaDB.

Run it to index a PDF:
    python src\embed.py "data\yourfile.pdf"

Run it with no arguments to see what is already stored:
    python src\embed.py
"""

import sys
from pathlib import Path

import chromadb
import requests

from config import (CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE,
                    EMBED_MODEL, HEADER_INCLUDE_FILENAME, OLLAMA_URL,
                    USE_CONTEXT_HEADERS)
from ingest import load_and_chunk, load_pdf_page_chunks

COLLECTION_NAME = "documents"


def index_config_metadata(chunk_size=None, overlap=None, headers=None,
                          header_include_filename=None,
                          embedding_model=None):
    """Describe the settings that affect vectors stored in a collection."""
    return {
        "hnsw:space": "cosine",
        "index_schema": "sentinel-rag-v1",
        "embedding_model": embedding_model or EMBED_MODEL,
        "chunk_size": str(chunk_size if chunk_size is not None else CHUNK_SIZE),
        "chunk_overlap": str(overlap if overlap is not None else CHUNK_OVERLAP),
        "context_headers": str(headers if headers is not None else USE_CONTEXT_HEADERS).lower(),
        "header_include_filename": str(
            header_include_filename
            if header_include_filename is not None
            else HEADER_INCLUDE_FILENAME
        ).lower(),
    }


def collection_config_status(collection, chunk_size=None, overlap=None,
                             headers=None, header_include_filename=None,
                             embedding_model=None):
    """Compare persisted collection metadata with the active configuration."""
    metadata = getattr(collection, "metadata", None) or {}
    if not metadata:
        return {"status": "missing", "stored": {}, "expected": index_config_metadata(
            chunk_size, overlap, headers, header_include_filename, embedding_model
        )}

    expected = index_config_metadata(
        chunk_size, overlap, headers, header_include_filename, embedding_model
    )
    mismatches = {}
    for key in ("embedding_model", "chunk_size", "chunk_overlap",
                "context_headers", "header_include_filename"):
        stored = str(metadata.get(key, "")).lower()
        wanted = str(expected[key]).lower()
        if not stored:
            mismatches[key] = {"stored": None, "expected": wanted}
        elif stored != wanted:
            mismatches[key] = {"stored": stored, "expected": wanted}
    if mismatches:
        return {"status": "incompatible", "stored": metadata, "expected": expected,
                "mismatches": mismatches}
    return {"status": "compatible", "stored": metadata, "expected": expected}


def set_collection_config(collection, chunk_size=None, overlap=None,
                          headers=None, header_include_filename=None,
                          embedding_model=None):
    """Persist the active index settings after a successful indexing pass."""
    metadata = index_config_metadata(
        chunk_size, overlap, headers, header_include_filename, embedding_model
    )
    metadata.pop("hnsw:space")
    existing = getattr(collection, "metadata", None) or {}
    metadata = {
        **{key: value for key, value in existing.items()
           if key != "hnsw:space"},
        **metadata,
    }
    if hasattr(collection, "modify"):
        collection.modify(metadata=metadata)


def incompatible_collection_error(status, collection_name):
    mismatches = ", ".join(status.get("mismatches", {}))
    return RuntimeError(
        f"Collection '{collection_name}' has an incompatible index "
        f"(mismatches: {mismatches or 'missing index metadata'}). "
        "Rebuild the collection after changing embedding or chunking settings."
    )


def embed_text(text):
    """Turn one piece of text into a vector using Ollama.

    An embedding is a list of numbers that encodes MEANING. Texts with similar
    meaning produce vectors that point in similar directions, which is what
    makes semantic search possible: we can find relevant text without any
    keyword overlap at all.

    nomic-embed-text returns 768 numbers per text.
    """
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=120,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_URL}. Is it running?\n"
            "Start it with:  ollama serve"
        ) from None

    if response.status_code != 200:
        # raise_for_status() hides the server's explanation. Ollama returns a
        # JSON body like {"error": "..."} that says what actually went wrong -
        # usually a bad model name or not enough memory. Always surface it.
        raise RuntimeError(
            f"Ollama returned {response.status_code} from /api/embeddings\n"
            f"Model requested: {EMBED_MODEL!r}\n"
            f"Server said: {response.text.strip()}"
        )

    return response.json()["embedding"]


def get_collection(name=COLLECTION_NAME, chunk_size=None, overlap=None,
                   headers=None, header_include_filename=None,
                   embedding_model=None):
    """Open (or create) a persistent ChromaDB collection.

    The name is a parameter so Day 5 can build several collections with
    different chunking settings and compare them side by side, without
    destroying the main index.

    PersistentClient writes to disk, so the index survives restarting the
    script. Embedding is by far the slowest step in the project - you never
    want to redo it just because you closed a terminal.

    hnsw:space="cosine" is important. Chroma defaults to squared L2 distance;
    we want COSINE similarity, which compares the ANGLE between vectors and
    therefore ignores text length. Two passages about the same topic should
    match whether one is 50 words and the other 400.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=name,
        metadata=index_config_metadata(
            chunk_size, overlap, headers, header_include_filename,
            embedding_model
        ),
    )


def has_page_metadata(collection):
    """Whether every stored chunk has current physical and printed page keys."""
    if not collection.count():
        return True
    data = collection.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []
    return bool(metadatas) and all(
        meta and "pdf_page" in meta and "printed_page" in meta
        for meta in metadatas)


def index_pdf(pdf_path, collection_name=COLLECTION_NAME,
              chunk_size=None, overlap=None, headers=None, verbose=True,
              system="general", source_type="reference", car_model="",
              source_hash="", document_id=None, mark_collection_config=False):
    """Full indexing pipeline: PDF -> chunks -> vectors -> stored in Chroma."""
    from config import (CHUNK_OVERLAP, CHUNK_SIZE, HEADER_INCLUDE_FILENAME,
                       USE_CONTEXT_HEADERS)
    from ingest import add_context_headers

    chunk_size = chunk_size or CHUNK_SIZE
    overlap = CHUNK_OVERLAP if overlap is None else overlap
    headers = USE_CONTEXT_HEADERS if headers is None else headers

    pdf_path = Path(pdf_path)
    doc_id = document_id or pdf_path.stem
    page_chunks = load_pdf_page_chunks(pdf_path, chunk_size=chunk_size,
                                       overlap=overlap, verbose=verbose)
    if not page_chunks:
        raise ValueError(f"No text could be extracted from '{pdf_path}'.")
    chunks = [item["text"] for item in page_chunks]

    if headers:
        chunks = add_context_headers(chunks, pdf_path.name,
                                     include_filename=HEADER_INCLUDE_FILENAME)
        if verbose:
            print("  -> context headers added; first chunk now begins:")
            print(f"     {chunks[0].splitlines()[0]}")

    collection = get_collection(
        collection_name,
        chunk_size=chunk_size,
        overlap=overlap,
        headers=headers,
        header_include_filename=HEADER_INCLUDE_FILENAME,
        embedding_model=EMBED_MODEL,
    )

    old = collection.get(where={"source": pdf_path.name},
                         include=["metadatas"])
    old_ids = set(old.get("ids") or [])

    if verbose:
        print(f"\nEmbedding {len(chunks)} chunks with {EMBED_MODEL}...")
    ids, vectors, metadatas = [], [], []

    for i, chunk in enumerate(chunks):
        # Deterministic IDs: re-indexing the same PDF OVERWRITES the old
        # entries instead of creating duplicate copies. Duplicate chunks are a
        # nasty bug - they crowd out genuinely different results in top-k.
        ids.append(f"{doc_id}::{i}")
        try:
            vectors.append(embed_text(chunk))
        except RuntimeError as exc:
            raise RuntimeError(
                f"Embedding failed for {pdf_path.name}, chunk {i + 1}/"
                f"{len(chunks)} on PDF page {page_chunks[i]['pdf_page']}: {exc}"
            ) from exc
        metadatas.append({"source": pdf_path.name,
                          "source_path": str(pdf_path.resolve()),
                          "chunk_index": i,
                          "page": page_chunks[i]["pdf_page"],
                          "pdf_page": page_chunks[i]["pdf_page"],
                          # ChromaDB metadata values cannot be Python None;
                          # empty string is normalized back to None on read.
                          "printed_page": page_chunks[i].get("printed_page") or "",
                          "system": system, "source_type": source_type,
                          "car_model": car_model,
                          "source_hash": source_hash or "unknown"})
        if verbose:
            print(f"  chunk {i + 1}/{len(chunks)} embedded "
                  f"({len(vectors[-1])} dimensions)")

    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=chunks,
        metadatas=metadatas,
    )

    stale_ids = old_ids.difference(ids)
    if stale_ids:
        collection.delete(ids=list(stale_ids))
    if mark_collection_config:
        records = collection.get(include=["metadatas"])
        sources = {
            meta.get("source") for meta in (records.get("metadatas") or [])
            if meta and meta.get("source")
        }
        if sources <= {pdf_path.name}:
            set_collection_config(
                collection,
                chunk_size=chunk_size,
                overlap=overlap,
                headers=headers,
                header_include_filename=HEADER_INCLUDE_FILENAME,
                embedding_model=EMBED_MODEL,
            )

    if verbose:
        print(f"\nStored {len(chunks)} chunks in {CHROMA_DIR}")
        print(f"Collection '{collection_name}' now holds "
              f"{collection.count()} chunks total")
    return collection


def cosine_similarity(a, b):
    """Similarity between two vectors, from -1 (opposite) to 1 (identical).

    Written out by hand rather than imported, so the concept is concrete:
    it is the dot product divided by the product of the two lengths, which
    is exactly the cosine of the angle between the vectors.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


def sanity_test():
    """Prove the embeddings actually encode meaning.

    Do not skip this. If these numbers come out wrong, everything built on top
    of them is worthless, and the failure is silent - you would only notice it
    on Day 4 as mysteriously bad answers.
    """
    print("\n" + "=" * 70)
    print("SANITY TEST - do similar texts really produce closer vectors?")
    print("=" * 70)

    a = "The Toubkal supercomputer has GPU nodes for scientific computing."
    b = "Toubkal's cluster includes NVIDIA accelerators for HPC workloads."
    c = "I baked a chocolate cake for my sister's birthday party."

    va, vb, vc = embed_text(a), embed_text(b), embed_text(c)

    sim_ab = cosine_similarity(va, vb)
    sim_ac = cosine_similarity(va, vc)

    print(f"\nA: {a}")
    print(f"B: {b}")
    print(f"C: {c}")
    print(f"\nsimilarity(A, B) = {sim_ab:.4f}   <- same topic, different words")
    print(f"similarity(A, C) = {sim_ac:.4f}   <- unrelated topic")

    if sim_ab > sim_ac:
        print("\nPASS - the embedding model encodes meaning, not vocabulary.")
        print("Note that A and B share almost no words, yet score high.")
    else:
        print("\nFAIL - something is wrong with the embedding setup.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        index_pdf(sys.argv[1], mark_collection_config=True)
        sanity_test()
    else:
        collection = get_collection()
        print(f"Collection '{COLLECTION_NAME}' holds "
              f"{collection.count()} chunks.")
        if collection.count():
            peek = collection.peek(limit=3)
            for doc_id, meta in zip(peek["ids"], peek["metadatas"]):
                print(f"  {doc_id}  <- {meta['source']}")
