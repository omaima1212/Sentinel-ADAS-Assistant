"""Shared page-reference formatting for retrieved PDF chunks."""


def page_reference(item):
    """Return the preferred printed page reference, or the PDF page fallback."""
    printed_page = item.get("printed_page")
    if printed_page not in (None, ""):
        return f"p. {printed_page}"
    pdf_page = item.get("pdf_page") or item.get("page")
    return f"PDF p. {pdf_page}" if pdf_page is not None else None


def source_citation(item):
    """Format a source and its page reference for answers and source panels."""
    source = item.get("source") or "Document"
    reference = page_reference(item)
    return f"{source}, {reference}" if reference else source


def citation_line(items):
    """Return one compact, deduplicated source line for a set of retrieved chunks."""
    citations = []
    for item in items:
        if not item.get("source"):
            continue
        citation = f"({source_citation(item)})"
        if citation not in citations:
            citations.append(citation)
    return "Sources: " + "; ".join(citations) if citations else ""
