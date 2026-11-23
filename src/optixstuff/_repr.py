"""Shared formatting helper for hierarchical ``__repr__`` methods.

Each leaf class in optixstuff (primary, detector, throughput elements,
coronagraphs) defines its own one-line ``__repr__``; the
:class:`Instrument` aggregator uses :func:`indent` to nest the
children one level deeper. Primary summaries label both diameter conventions;
coronagraph summaries distinguish data extent from a design working angle.
"""

from __future__ import annotations


def indent(text: str, prefix: str = "  ") -> str:
    """Prefix every line of ``text`` with ``prefix``.

    Args:
        text: Text to indent, possibly spanning several lines.
        prefix: String placed at the start of every line.

    Returns:
        The text with every line prefixed.
    """
    return "\n".join(prefix + line for line in text.split("\n"))
