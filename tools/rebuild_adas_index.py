"""Re-embed every locally available source in the shared ADAS collection.

Run after changing embedding or chunking settings, or when upgrading a legacy
collection that has no recorded index configuration.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import ADAS_COLLECTION_NAME, ADAS_DATA_DIR
from embed import get_collection, index_pdf, set_collection_config

ADDITIONAL_DIR = ROOT / "additional_adas_knowledge"


def _sources():
    primary_metadata_path = ADAS_DATA_DIR / "metadata.json"
    primary_metadata = (json.loads(primary_metadata_path.read_text(encoding="utf-8"))
                        if primary_metadata_path.exists() else {})
    additional_metadata_path = ADDITIONAL_DIR / "metadata.json"
    additional_metadata = json.loads(additional_metadata_path.read_text(encoding="utf-8"))
    sources = []

    for path in sorted(ADAS_DATA_DIR.glob("*.pdf")):
        labels = primary_metadata.get(path.name, {})
        sources.append((path, path.name, labels, path.name))
    for path in sorted(ADDITIONAL_DIR.rglob("*.pdf")):
        relative = path.relative_to(ADDITIONAL_DIR).as_posix()
        labels = additional_metadata.get(relative)
        if not isinstance(labels, dict):
            raise ValueError(f"Missing ADAS metadata for {relative}")
        sources.append((path, relative, labels, path.name))

    names = [source_name for _, _, _, source_name in sources]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            "Source filenames must be unique in the shared ADAS collection: "
            + ", ".join(duplicates)
        )
    return sources


def rebuild_adas_index():
    collection = get_collection(ADAS_COLLECTION_NAME)
    sources = _sources()
    if not sources:
        raise RuntimeError("No ADAS PDFs were found to index.")

    for path, key, labels, source_name in sources:
        old = collection.get(where={"source": source_name},
                            include=["metadatas"])
        old_ids = old.get("ids") or []
        document_id = (old_ids[0].split("::", 1)[0] if old_ids
                       else hashlib.sha256(key.encode("utf-8")).hexdigest()[:16])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        index_pdf(
            path,
            collection_name=ADAS_COLLECTION_NAME,
            system=str(labels.get("system", "general")),
            source_type=str(labels.get("source_type", "reference")),
            car_model=str(labels.get("car_model", "")),
            source_hash=digest,
            document_id=document_id,
        )
        current = collection.get(where={"source": source_name},
                                 include=["metadatas"])
        updated = []
        for metadata in current.get("metadatas") or []:
            item = dict(metadata or {})
            item.update(labels)
            updated.append(item)
        if updated:
            collection.update(ids=current["ids"], metadatas=updated)
        print(f"Rebuilt {source_name}: {len(current.get('ids') or [])} chunks")

    stored = collection.get(include=["metadatas"])
    indexed_sources = {
        metadata.get("source") for metadata in (stored.get("metadatas") or [])
        if metadata and metadata.get("source")
    }
    expected_sources = {source_name for _, _, _, source_name in sources}
    if indexed_sources != expected_sources:
        raise RuntimeError(
            "Collection contains missing or unrecognized sources; index metadata "
            "was not updated. Reconcile the collection before querying."
        )

    set_collection_config(collection)
    print(f"ADAS collection '{ADAS_COLLECTION_NAME}' is rebuilt "
          f"({collection.count()} chunks).")
    return collection.count()


if __name__ == "__main__":
    rebuild_adas_index()
