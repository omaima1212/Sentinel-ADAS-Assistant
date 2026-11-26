"""Safely append supplemental ADAS PDFs to the existing ADAS collection.

This standalone indexer reads PDFs under additional_adas_knowledge and their
labels from that directory's metadata.json. It never deletes collection data.
If a filename is already used by another source, it refuses that PDF.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import ADAS_COLLECTION_NAME
from embed import collection_config_status, get_collection, index_pdf

DOCS_DIR = ROOT / "additional_adas_knowledge"
METADATA_FILE = DOCS_DIR / "metadata.json"


def index_additional_documents():
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("additional_adas_knowledge/metadata.json must be an object")

    collection = get_collection(ADAS_COLLECTION_NAME)
    if collection_config_status(collection)["status"] != "compatible":
        raise RuntimeError(
            "The ADAS collection uses legacy or incompatible index settings. "
            "Run `python tools\\rebuild_adas_index.py` to rebuild all ADAS "
            "sources before indexing incrementally."
        )
    indexed, skipped, conflicts, failed = [], [], [], []
    for pdf_path in sorted(DOCS_DIR.rglob("*.pdf")):
        relative = pdf_path.relative_to(DOCS_DIR).as_posix()
        labels = metadata.get(relative)
        if not isinstance(labels, dict):
            failed.append((relative, "missing metadata entry"))
            continue

        digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        source_name = pdf_path.name
        existing = collection.get(where={"source": source_name}, include=["metadatas"])
        old_ids = existing.get("ids") or []
        old_metas = existing.get("metadatas") or []
        if old_metas and any(
            Path(str(meta.get("source_path", ""))).resolve() != pdf_path.resolve()
            for meta in old_metas
        ):
            conflicts.append((relative, "same source filename belongs to another indexed PDF"))
            continue

        if old_metas and all(meta.get("source_hash") == digest for meta in old_metas):
            # Refresh supplemental metadata without re-embedding unchanged text.
            merged = []
            for old in old_metas:
                item = dict(old)
                item.update(labels)
                merged.append(item)
            collection.update(ids=old_ids, metadatas=merged)
            skipped.append(relative)
            continue

        document_id = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]
        try:
            index_pdf(
                pdf_path,
                collection_name=ADAS_COLLECTION_NAME,
                system=str(labels.get("system", "general")),
                source_type=str(labels.get("source_type", "official_reference")),
                car_model="",
                source_hash=digest,
                document_id=document_id,
                verbose=False,
            )
            current = collection.get(where={"source": source_name}, include=["metadatas"])
            updated = []
            for item in current.get("metadatas") or []:
                row = dict(item)
                row.update(labels)
                updated.append(row)
            if updated:
                collection.update(ids=current["ids"], metadatas=updated)
            indexed.append((relative, len(current.get("ids") or [])))
        except Exception as exc:
            failed.append((relative, str(exc)))

    print(f"ADAS collection: {ADAS_COLLECTION_NAME}")
    print(f"Collection chunks now: {collection.count()}")
    print(f"Indexed: {len(indexed)} | unchanged: {len(skipped)} | "
          f"conflicts: {len(conflicts)} | failed: {len(failed)}")
    for path, count in indexed:
        print(f"  indexed {path}: {count} chunks")
    for group, rows in (("unchanged", skipped), ("conflict", conflicts), ("failed", failed)):
        for row in rows:
            if isinstance(row, tuple):
                path, reason = row
            else:
                path, reason = row, "unchanged source hash and metadata"
            print(f"  {group} {path}: {reason}")
    return {"indexed": indexed, "skipped": skipped, "conflicts": conflicts,
            "failed": failed, "collection_count": collection.count()}


if __name__ == "__main__":
    index_additional_documents()
