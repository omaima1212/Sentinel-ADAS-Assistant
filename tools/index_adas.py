"""Idempotently index data/adas/*.pdf into the shared ADAS Chroma collection.

Add PDFs and a matching metadata.json entry, then run:
    python tools/index_adas.py
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import ADAS_COLLECTION_NAME, ADAS_DATA_DIR, ADAS_METADATA_FILE
from embed import (collection_config_status, get_collection, index_pdf)


def index_adas_documents():
    ADAS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ADAS_METADATA_FILE.exists():
        ADAS_METADATA_FILE.write_text("{}\n", encoding="utf-8")
    metadata = json.loads(ADAS_METADATA_FILE.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must be an object keyed by PDF filename")

    collection = get_collection(ADAS_COLLECTION_NAME)
    if collection_config_status(collection)["status"] != "compatible":
        raise RuntimeError(
            "The ADAS collection uses legacy or incompatible index settings. "
            "Run `python tools\\rebuild_adas_index.py` to rebuild all ADAS "
            "sources before indexing incrementally."
        )
    pdfs = sorted(ADAS_DATA_DIR.glob("*.pdf"))
    indexed, skipped, failed = [], [], []
    for pdf_path in pdfs:
        labels = metadata.get(pdf_path.name, {})
        if not isinstance(labels, dict):
            failed.append((pdf_path.name, "metadata entry must be an object"))
            continue
        labels = {
            "system": str(labels.get("system", "general")),
            "source_type": str(labels.get("source_type", "reference")),
            "car_model": str(labels.get("car_model", "")),
        }
        digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        existing = collection.get(where={"source": pdf_path.name},
                                  include=["metadatas"])
        old_metadata = existing.get("metadatas") or []
        missing_printed_page = any(
            not item or "printed_page" not in item or "pdf_page" not in item
            for item in old_metadata)
        if old_metadata and all(
            item.get("source_hash") == digest
            and all(item.get(key) == value for key, value in labels.items())
            for item in old_metadata
        ) and not missing_printed_page:
            skipped.append(pdf_path.name)
            continue

        if missing_printed_page:
            print(f"Re-indexing {pdf_path.name}: stored chunks lack printed_page metadata.")
        if existing.get("ids"):
            collection.delete(where={"source": pdf_path.name})
        document_id = hashlib.sha256(pdf_path.name.encode("utf-8")).hexdigest()[:16]
        try:
            index_pdf(pdf_path, collection_name=ADAS_COLLECTION_NAME,
                      system=labels["system"], source_type=labels["source_type"],
                      car_model=labels["car_model"], source_hash=digest,
                      document_id=document_id, verbose=False)
            indexed.append(pdf_path.name)
        except Exception as exc:
            failed.append((pdf_path.name, str(exc)))

    print(f"ADAS collection: {ADAS_COLLECTION_NAME}")
    print(f"PDFs found: {len(pdfs)} | indexed/updated: {len(indexed)} | "
          f"unchanged: {len(skipped)} | failed: {len(failed)}")
    for filename in indexed:
        print(f"  indexed  {filename}")
    for filename in skipped:
        print(f"  current  {filename}")
    for filename, error in failed:
        print(f"  FAILED   {filename}: {error}")
    if not pdfs:
        print(f"Add PDFs to {ADAS_DATA_DIR} and label them in {ADAS_METADATA_FILE}.")
    return {"indexed": indexed, "skipped": skipped, "failed": failed,
            "collection_count": collection.count()}


if __name__ == "__main__":
    index_adas_documents()
