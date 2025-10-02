"""Utility helpers for text normalization and identifiers."""
from __future__ import annotations

import hashlib
import re


def slugify_document_title(name: str) -> str:
    """Create a compact ASCII-friendly identifier that stays stable for a title."""
    hash_hex = hashlib.md5(name.encode("utf-8")).hexdigest()[:12]

    ascii_part = re.sub(r"[^\w]", "_", name.lower(), flags=re.ASCII)
    ascii_part = re.sub(r"_+", "_", ascii_part).strip("_")

    if not ascii_part or len(ascii_part) > 20:
        return f"doc_{hash_hex}"

    return f"{ascii_part}_{hash_hex}"


def parse_section_code(section_title: str) -> str:
    """Extract numeric prefix like "1.1" from a section heading if present."""
    match = re.match(r"^(\d+(?:\.\d+)*)", section_title.strip())
    return match.group(1) if match else ""


__all__ = [
    "slugify_document_title",
    "parse_section_code",
]
