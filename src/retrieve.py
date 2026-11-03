"""
Day 4 - Stage 3 of the pipeline: question -> the most relevant chunks.

This is the "R" in RAG, and it is where most of the quality of the whole
system lives. If the right chunk is not retrieved here, nothing downstream
can recover it.

Run it on its own to inspect retrieval WITHOUT any model involved:
    python src\retrieve.py "what supercomputer did you work on?"
"""

import sys
import re
import hashlib
from pathlib import Path

from config import (ADAS_DATA_DIR, EXPAND_NEIGHBORS, HYBRID_CANDIDATE_K,
                    HYBRID_RRF_K, PROJECT_ROOT, TOP_K, USE_HYBRID)
from citations import page_reference
from embed import embed_text, get_collection, index_pdf


def retrieve(question, k=TOP_K, collection_name=None, system=None,
             car_model=None, use_hybrid=None, _allow_reindex=True):
    """Find the top-k matches, with up to two adjacent context chunks.

    Three steps:
      1. Embed the question with the SAME model used to index the documents.
         (Different models = unrelated coordinate spaces = meaningless scores.)
      2. Ask Chroma for the k nearest stored vectors.
      3. Return the TEXT filed alongside those vectors.

    Note step 3: the vector is only an index. The text is the payload.
    """
    collection = (get_collection(collection_name) if collection_name
                  else get_collection())

    if collection.count() == 0:
        raise RuntimeError(
            "The collection is empty. Index a PDF first:\n"
            '    python src\\embed.py "data\\yourfile.pdf"'
        )

    question_vector = embed_text(question)

    filters = []
    if system:
        filters.append({"system": system})
    if car_model:
        filters.append({"car_model": car_model})
    where = ({"$and": filters} if len(filters) > 1 else
             filters[0] if filters else None)
    # Chroma errors when the requested filter has no matches; report that as
    # an empty retrieval so callers can give a grounded refusal.
    hybrid = USE_HYBRID if use_hybrid is None else use_hybrid
    try:
        results = collection.query(
            query_embeddings=[question_vector],
            n_results=min(max(k, HYBRID_CANDIDATE_K) if hybrid else k,
                          collection.count()),
            include=["documents", "metadatas", "distances"],
            where=where,
        )
    except Exception:
        if filters:
            return []
        raise

    # Chroma returns a list-per-query; we only sent one query, so take index 0.
    hits = []
    for text, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # With hnsw:space="cosine", Chroma reports cosine DISTANCE.
        # Convert to the similarity score we reason about: 1 = identical.
        hits.append({
            "text": text,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "page": meta.get("pdf_page", meta.get("page")),
            "pdf_page": meta.get("pdf_page", meta.get("page")),
            "printed_page": meta.get("printed_page") or None,
            "_legacy_page_metadata": (not meta or "printed_page" not in meta
                                       or "pdf_page" not in meta),
            "system": meta.get("system", "general"),
            "source_type": meta.get("source_type", "reference"),
            "car_model": meta.get("car_model", ""),
            "similarity": 1 - distance,
            "neighbor": False,
        })

    if hybrid and hits:
        hits = _reciprocal_rank_fusion(question, collection, hits, k, where)
    else:
        hits = hits[:k]

    metadata_to_check = list(results["metadatas"][0]) + [
        {"source": hit["source"]} for hit in hits
        if hit.get("_legacy_page_metadata")
    ]
    if _allow_reindex and reindex_missing_page_metadata(
            collection, collection_name, metadata_to_check):
        return retrieve(question, k=k, collection_name=collection_name,
                        system=system, car_model=car_model,
                        use_hybrid=use_hybrid, _allow_reindex=False)
    for hit in hits:
        hit.pop("_legacy_page_metadata", None)
    if EXPAND_NEIGHBORS:
        hits = _expand_neighbors(collection, hits, k, collection_name)
    return hits


def reindex_missing_page_metadata(collection, collection_name, raw_metas):
    """Re-index a locally available PDF when retrieved metadata is obsolete."""
    sources = {meta.get("source") for meta in raw_metas
               if meta and ("printed_page" not in meta or "pdf_page" not in meta)}
    sources.discard(None)
    if not sources:
        return False

    reindexed = False
    for source in sources:
        records = collection.get(where={"source": source}, include=["metadatas"])
        metadatas = records.get("metadatas") or []
        if metadatas and all(meta and "printed_page" in meta and "pdf_page" in meta
                             for meta in metadatas):
            continue
        first_meta = next((meta for meta in metadatas if meta), {})
        path = _find_source_pdf(source, first_meta)
        if path is None:
            raise RuntimeError(
                f"Stored chunks for '{source}' lack printed_page metadata. "
                "The PDF was not found locally, so re-index it before querying."
            )
        print(f"Re-indexing {source}: stored chunks lack printed_page metadata.")
        ids = records.get("ids") or []
        document_id = (ids[0].split("::", 1)[0] if ids else path.stem)
        source_hash = first_meta.get("source_hash")
        if not source_hash or source_hash == "unknown":
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        index_pdf(
            path, collection_name=collection_name or "documents",
            system=first_meta.get("system", "general"),
            source_type=first_meta.get("source_type", "reference"),
            car_model=first_meta.get("car_model", ""),
            source_hash=source_hash,
            document_id=document_id, verbose=False,
        )
        reindexed = True
    return reindexed


def _find_source_pdf(source, metadata):
    candidates = []
    stored_path = metadata.get("source_path")
    if stored_path:
        candidates.append(Path(stored_path))
    candidates.extend([
        Path(source),
        PROJECT_ROOT / "data" / source,
        ADAS_DATA_DIR / source,
        PROJECT_ROOT / "uploads" / source,
    ])
    return next((path for path in candidates if path.is_file()), None)


def _expand_neighbors(collection, hits, k, collection_name=None):
    """Add up to two adjacent chunks from the first two ranked hits."""
    max_total = TOP_K + 2
    hits = list(hits[:min(k, max_total)])
    expanded = {(hit["source"], hit["chunk_index"]): dict(hit) for hit in hits}
    additions = []
    neighbor_slots = max(0, max_total - len(hits))
    if not neighbor_slots:
        return sorted(expanded.values(),
                      key=lambda hit: (hit["source"], hit["chunk_index"]))
    for anchor in hits[:2]:
        source = anchor["source"]
        chunk_index = int(anchor["chunk_index"])
        for neighbor_index in (chunk_index - 1, chunk_index + 1):
            key = (source, neighbor_index)
            if neighbor_index < 0 or key in expanded:
                continue
            result = collection.get(
                where={"$and": [{"source": source},
                                {"chunk_index": neighbor_index}]},
                include=["documents", "metadatas"],
            )
            ids = result.get("ids") or []
            docs = result.get("documents") or []
            metas = result.get("metadatas") or []
            if not ids or not docs:
                continue
            meta = metas[0] or {}
            if "printed_page" not in meta or "pdf_page" not in meta:
                reindex_missing_page_metadata(
                    collection, collection_name or getattr(collection, "name", "documents"),
                    [meta or {"source": source}],
                )
                result = collection.get(
                    where={"$and": [{"source": source},
                                    {"chunk_index": neighbor_index}]},
                    include=["documents", "metadatas"],
                )
                ids = result.get("ids") or []
                docs = result.get("documents") or []
                metas = result.get("metadatas") or []
                if not ids or not docs:
                    continue
                meta = metas[0] or {}
            addition = {
                "text": docs[0], "source": source,
                "chunk_index": meta.get("chunk_index", neighbor_index),
                "page": meta.get("pdf_page", meta.get("page")),
                "pdf_page": meta.get("pdf_page", meta.get("page")),
                "printed_page": meta.get("printed_page") or None,
                "system": meta.get("system", anchor.get("system", "general")),
                "source_type": meta.get("source_type", "reference"),
                "car_model": meta.get("car_model", ""),
                "similarity": None, "neighbor": True,
            }
            expanded[key] = addition
            additions.append(key)
            if len(additions) >= neighbor_slots:
                break
        if len(additions) >= neighbor_slots:
            break

    chosen = list(expanded.values())
    chosen.sort(key=lambda hit: (hit["source"], hit["chunk_index"]))
    return chosen[:max_total]


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _reciprocal_rank_fusion(question, collection, vector_hits, k, where):
    """Fuse vector and BM25 rankings while retaining original source metadata."""
    from rank_bm25 import BM25Okapi

    corpus = collection.get(where=where,
                            include=["documents", "metadatas"])
    docs = corpus.get("documents") or []
    metas = corpus.get("metadatas") or []
    if not docs:
        return vector_hits[:k]
    bm25 = BM25Okapi([_tokens(doc or "") for doc in docs])
    query_tokens = _tokens(question)
    scores = bm25.get_scores(query_tokens)
    query_terms = set(query_tokens)
    # Okapi can assign negative IDF to terms in very small corpora. Use actual
    # token overlap to decide whether a result is eligible, then let BM25 rank
    # those candidates by score.
    matched = [i for i, doc in enumerate(docs)
               if query_terms.intersection(_tokens(doc or ""))]
    bm25_order = sorted(matched, key=lambda i: scores[i], reverse=True)

    fused, records = {}, {}
    for rank, hit in enumerate(vector_hits, start=1):
        key = f"{hit['source']}::{hit['chunk_index']}"
        fused[key] = fused.get(key, 0.0) + 1 / (HYBRID_RRF_K + rank)
        records[key] = hit
    for rank, idx in enumerate(bm25_order[:HYBRID_CANDIDATE_K], start=1):
        meta = metas[idx] or {}
        key = f"{meta.get('source', 'unknown')}::{meta.get('chunk_index', idx)}"
        fused[key] = fused.get(key, 0.0) + 1 / (HYBRID_RRF_K + rank)
        records.setdefault(key, {
            "text": docs[idx], "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", idx),
            "page": meta.get("pdf_page", meta.get("page")),
            "system": meta.get("system", "general"),
            "pdf_page": meta.get("pdf_page", meta.get("page")),
            "printed_page": meta.get("printed_page") or None,
            "source_type": meta.get("source_type", "reference"),
            "car_model": meta.get("car_model", ""), "similarity": 0.0,
            "neighbor": False,
            "_legacy_page_metadata": (not meta or "printed_page" not in meta
                                       or "pdf_page" not in meta),
        })
    ranked = sorted(fused, key=fused.get, reverse=True)[:k]
    output = []
    for key in ranked:
        hit = dict(records[key])
        hit["rrf_score"] = fused[key]
        output.append(hit)
    return output


def print_hits(hits, preview_chars=400):
    """Display retrieved chunks for inspection.

    Get in the habit of looking at this BEFORE looking at the model's answer.
    Most bad RAG answers are retrieval failures wearing a generation-failure
    costume - the model was simply never shown the right passage.
    """
    print(f"\nRetrieved {len(hits)} chunks:\n")
    for rank, hit in enumerate(hits, start=1):
        print("=" * 70)
        similarity = hit.get("similarity")
        score = (f"similarity {similarity:.4f}" if similarity is not None
                 else "neighbor")
        print(f"#{rank}  {score}  "
              f"| chunk {hit['chunk_index']} of {hit['source']}")
        reference = page_reference(hit)
        if reference:
            print(f"    {reference}")
        print("=" * 70)
        print(hit["text"][:preview_chars].strip())
        if len(hit["text"]) > preview_chars:
            print("   [...]")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src\\retrieve.py "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"Question: {question}")
    print_hits(retrieve(question))

    # MEASURED on the internship logbook with nomic-embed-text (Day 5).
    # An earlier guess here said "> 0.75 = strong match". That was wrong: no
    # correct answer in the whole evaluation scored above 0.65.
    print("Measured score ranges for this document + model:")
    print("  correct answers   0.49 - 0.64")
    print("  correct refusals  0.36 - 0.43")
    print()
    print("Do NOT threshold on these numbers. A correct hit scored 0.4923")
    print("while an irrelevant one scored 0.4291 - a gap of 0.06. Absolute")
    print("cosine values are not calibrated across documents or models;")
    print("only the RELATIVE ordering within a single query is meaningful.")
