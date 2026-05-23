"""Shared formatting helper for hierarchical ``__repr__`` methods.

Each leaf class in optixstuff (primary, detector, throughput elements,
coronagraphs) defines its own one-line ``__repr__``; the
:class:`OpticalPath` aggregator uses :func:`indent` to nest the
children one level deeper. Same pattern as ``skyscapes._repr``.
"""

from __future__ import annotations


def indent(text: str, prefix: str = "  ") -> str:
    """Prefix every line of ``text`` with ``prefix``."""
    return "\n".join(prefix + line for line in text.split("\n"))
