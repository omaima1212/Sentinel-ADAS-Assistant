# Supplemental ADAS knowledge

This directory contains official Euro NCAP PDFs downloaded for the ADAS RAG
collection. The source URLs, category, version, and publication date are
recorded in `metadata.json`. Both PDFs were checked for extractable text. The
Safe Driving protocol was indexed. The AEB Car-to-Car protocol extracted
successfully, but embedding it failed because Ollama reported that an input
exceeded the embedding model's context length. Its current chunks were left
out of ChromaDB; the existing chunking settings were not changed.

Four other approved PDFs could not be retrieved because the official hosts
returned empty responses, 403, or 404 errors. They were not fabricated or
indexed:

| Intended document | Official source | Retrieval issue |
|---|---|---|
| UN Regulation No. 152, 2024/2497 | [EUR-Lex PDF](https://eur-lex.europa.eu/eli/reg/2024/2497/oj/eng/pdf) | Empty response; UNECE fallback returned 403 |
| UN Regulation No. 171, 2024/2689 | [EUR-Lex PDF](https://eur-lex.europa.eu/eli/reg/2024/2689/oj/eng/pdf) | Empty response; UNECE fallback returned 403 |
| Euro NCAP LSS Test Protocol v4.3 | [Safety Assist protocols](https://www.euroncap.com/safety-assist/) | Current direct file URL returned 404 |
| Euro NCAP Collision Avoidance Assessment v10.4.1 | [Safety Assist protocols](https://www.euroncap.com/safety-assist/) | Official page host blocked the file request; CDN URL returned 404 |

## Documents present

| File | Category | Source | Version / date |
|---|---|---|---|
| `01_General_ADAS/Euro_NCAP_Safe_Driving_Assessment_v10.4.pdf` | General ADAS / driver monitoring | [Euro NCAP PDF](https://cdn.euroncap.com/cars/assets/euro_ncap_assessment_protocol_sa_safe_driving_v104_3090e88253.pdf) | v10.4, February 2024 |
| `02_ADAS_Systems/Euro_NCAP_AEB_Car_to_Car_v4.3.1.pdf` | AEB / collision scenarios | [Euro NCAP PDF](https://cdn.euroncap.com/cars/assets/euro_ncap_aeb_c2c_test_protocol_v431_532926aad1.pdf) | v4.3.1, February 2024 |

## Indexing

After Ollama is running with the project's configured `nomic-embed-text`
embedding model, run from the project root:

```powershell
python tools\index_additional_adas.py
```

The supplemental indexer uses the existing Chroma path, `adas_documents`
collection, and configured embedding model. It does not clear or recreate the
collection. It skips unchanged PDFs and refuses to index a PDF if another
source already owns that filename. New vectors carry source URL, category,
publication date, version, and document name metadata. A document that exceeds
the embedding model's context limit is reported as failed; chunking settings
must be reviewed before retrying it.

## Retrieval check questions

See `evaluation_questions.json` for a 20-question checklist covering ADAS
concepts, systems, sensors, limitations, and driver responsibility. Questions
outside the two retrieved protocols are included as coverage targets and may
not yet retrieve a supporting passage.
