# Sentinel: ADAS Assistant

Sentinel is a local assistant for questions about Advanced Driver Assistance Systems (ADAS). It retrieves passages from indexed PDFs and can provide citations, stopping-distance estimates, charts, and quizzes. It runs language and embedding models through [Ollama](https://ollama.com/).

The project has two interfaces: a Streamlit app and a lightweight Flask app. Both use the same retrieval and answer code. The Flask page also supports the ADAS assistant, tutor mode, and uploaded-PDF document chat.

## What it can do

- Answer from indexed documents and show source passages and page references.
- Provide an ADAS assistant and tutor/quiz mode over the shared ADAS index.
- Chat with an uploaded PDF using the document-chat mode.
- Estimate stopping distance with a clearly labeled simplified calculation.
- Generate stopping-distance and ADAS timeline charts.
- Optionally search the web when document answers are insufficient. The configured DuckDuckGo backend sends the search query outside the machine; the feature is separate from the local document index.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/download)
- Enough memory for the selected local models (the default chat model is `llama3.2:3b`)

Pull the default chat and embedding models:

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

`qwen2.5:3b` is an optional model-comparison candidate used by `tools/check_tool_models.py`.

Create an environment and install the Python packages:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run the app

From the project root, start either interface:

```powershell
streamlit run src\app.py
```

or:

```powershell
python src\server.py
```

The Flask interface is available at `http://localhost:5000`. Ollama must be running locally. The Chroma vector database is stored under `chroma_db/` and is ignored by Git, so rebuild it after cloning or deleting it.

## Index documents

The repository includes PDFs in `data/adas/` and `additional_adas_knowledge/`. These are source material, not generated database files.

Index PDFs from `data/adas/` into the shared ADAS collection:

```powershell
python tools\index_adas.py
```

The optional supplemental collection uses its own metadata and indexer:

```powershell
python tools\index_additional_adas.py
```

Both indexers use the configured `nomic-embed-text` model. Use the same embedding model for indexing and querying. The supplemental indexer’s document list, provenance, and indexing notes are in [additional_adas_knowledge/README.md](additional_adas_knowledge/README.md).

To re-embed every primary and supplemental ADAS PDF after changing index settings, or to upgrade an older collection, run:

```powershell
python tools\rebuild_adas_index.py
```

The standalone retrieval command uses the `documents` collection by default. To query the shared ADAS collection explicitly, use:

```powershell
python src\retrieve.py --collection adas_documents "What does automatic emergency braking do?"
```

To index an individual PDF for document chat:

```powershell
python src\embed.py "path\to\document.pdf"
```

The PDF extractors need a text layer; scanned image-only PDFs require OCR, which this project does not provide.

## Useful commands

```powershell
python src\agent.py "When does AEB intervene?"
python src\retrieve.py "your question"
python src\generate.py "your question"
python src\evaluate.py "run label"
python src\eval_adas.py
python src\experiment.py "path\to\document.pdf"
```

Run the automated checks with:

```powershell
pytest
```

Evaluation reports and generated charts are written under `results/`. The ADAS questions in `questions_adas.json` are placeholders. `python src/eval_adas.py` refuses to produce an accuracy result until that file contains only verified, non-placeholder questions with correct source filenames and answers. This prevents a placeholder suite from being reported as measured accuracy.

## Configuration

Project settings live in [src/config.py](src/config.py). The default web fallback backend is DuckDuckGo. DuckDuckGo may rate-limit repeated searches. The fallback can be disabled in the UI where available; ADAS agent mode may select web search when its document search is insufficient.

Other notable settings include chunk size, overlap, number of retrieved passages, neighbor expansion, and hybrid semantic/keyword retrieval (enabled by default). Changing embedding-affecting settings requires rebuilding the affected index.

## Project structure

| Path | Purpose |
|---|---|
| `src/app.py` | Streamlit interface |
| `src/server.py` and `src/web/` | Flask interface and browser page |
| `src/agent.py` | ADAS assistant, tools, and answer policy |
| `src/ingest.py`, `src/embed.py`, `src/retrieve.py` | PDF ingestion, indexing, and retrieval |
| `src/generate.py` | Prompt building and model calls |
| `src/charts.py`, `src/citations.py`, `src/websearch.py` | Charts, citations, and optional web search |
| `data/adas/` | Primary ADAS PDFs and labels |
| `additional_adas_knowledge/` | Supplemental PDFs, metadata, and indexing guide |
| `data/timelines/` | Data used for the ADAS timeline chart |
| `tools/` | Indexing and document-generation utilities |
| `tests/` | Automated checks |
| `results/` | Saved evaluation reports and generated charts |
| `CONCEPTS.md` | Explanations of the project’s main concepts |
| `SPECIFICATION.md` | Original project specification |
| `BUILD_LOG.md` | Development history and measured experiments |
