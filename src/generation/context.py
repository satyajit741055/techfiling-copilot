"""
Context Formatting for Generation.

Converts retrieved chunks into numbered source blocks for the LLM prompt.

Design decisions:
- Use `text` field, not `embed_text` (cleaner for generation)
- Include `element_orders` for traceability to golden labels
- Structured <source> tags with metadata attributes
- Return both formatted context and source mapping for validation
"""

from __future__ import annotations

from typing import Any


def format_source(index: int, chunk: dict[str, Any]) -> str:
    """
    Format a single chunk as a numbered source block.

    Example output:
        <source id="S1" doc="AAPL_10-K_2025" section="Products and Services Performance"
                type="table" elements="[279]" table_id="AAPL_10-K_2025_t016">
        **Table**
        | Products | 2025 | 2024 |
        |---|---|---|
        | iPhone | $209,586 | $201,183 |
        ...
        </source>

    Args:
        index: 1-based index for source ID
        chunk: Chunk dictionary with text and metadata

    Returns:
        Formatted source block string

    Why include element_orders?
        - Enables citation verification against golden labels
        - Maps back to exact parsed elements in the original filing
        - Useful for debugging and traceability
    """
    source_id = f"S{index}"

    # Build metadata attributes
    attrs = [
        f'id="{source_id}"',
        f'doc="{chunk["doc_id"]}"',
    ]

    # Add section info
    section = chunk.get("section_title") or chunk.get("section") or "Unknown"
    attrs.append(f'section="{section}"')

    # Add type (text or table)
    chunk_type = chunk.get("type", "text")
    attrs.append(f'type="{chunk_type}"')

    # Add element_orders for traceability back to parsed elements
    element_orders = chunk.get("element_orders", [])
    if element_orders:
        attrs.append(f'elements="{element_orders}"')

    # Add table-specific metadata
    if chunk.get("table_id"):
        attrs.append(f'table_id="{chunk["table_id"]}"')
    if chunk.get("units"):
        attrs.append(f'units="{chunk["units"]}"')

    # Build the source block
    attr_str = " ".join(attrs)
    text = chunk.get("text", "")

    return f"<source {attr_str}>\n{text}\n</source>"


def format_context(results: list) -> tuple[str, list[dict[str, Any]]]:
    """
    Format retrieved results into context string and source mapping.

    Args:
        results: List of SearchResult objects or chunk dictionaries

    Returns:
        Tuple of:
            - context: Formatted string for the prompt
            - sources: List of source metadata for validation/tracing

    The sources list contains:
        - source_id: Prompt-level ID (S1, S2, ...)
        - chunk_id: Original chunk identifier
        - doc_id: Document identifier
        - section: Section name
        - section_title: Section title
        - type: Chunk type (text or table)
        - element_orders: List of parsed element indices
        - table_id: Table identifier (if applicable)
        - score: Reranker score
    """
    blocks = []
    sources = []

    for i, result in enumerate(results, start=1):
        # Handle both SearchResult objects and raw dictionaries
        chunk = result.chunk if hasattr(result, "chunk") else result
        score = float(result.score) if hasattr(result, "score") else None

        # Format for prompt
        blocks.append(format_source(i, chunk))

        # Track source metadata for validation and API response
        sources.append(
            {
                "source_id": f"S{i}",
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "section": chunk.get("section"),
                "section_title": chunk.get("section_title"),
                "type": chunk.get("type"),
                "element_orders": chunk.get("element_orders", []),
                "table_id": chunk.get("table_id"),
                "score": score,
                "text": chunk.get("text", ""),  # Include text for API response
                "breadcrumb": chunk.get("breadcrumb"),  # Full path for UI display
            }
        )

    context = "\n\n".join(blocks)
    return context, sources
