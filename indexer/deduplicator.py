"""
Safe deduplication for STM32 ParsedRegister records.

Goals:
1) Do NOT collapse across devices (that breaks device-aware ranking).
2) Only remove TRUE duplicates (usually from re-ingestion or duplicate SVD sources).
3) Optionally deduplicate long description strings into a canonical store (storage optimization).

This preserves:
- per-device identity (device, series)
- per-instance identity (peripheral name like DMA1 vs DMA2, I2C1 vs I2C4, etc.)
- correct per-device address context (full_address)
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import hashlib
import re

from .models import ParsedRegister


# ----------------------------
# Normalization helpers
# ----------------------------

_WS_RE = re.compile(r"\s+")


def _norm_text(s: Optional[str]) -> str:
    """
    Normalize free-text for hashing/deduping.
    Keeps meaning but collapses superficial differences.
    """
    if not s:
        return ""
    s = s.strip()
    s = _WS_RE.sub(" ", s)
    return s


def _hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ----------------------------
# Exact identity fingerprint
# ----------------------------

def _compute_exact_register_key(reg: ParsedRegister) -> str:
    """
    A strict identity key for TRUE duplicates only.

    Two registers are considered the same only if they are effectively the same
    extracted record from the same device + same peripheral instance + same register
    + same address + same field layout.

    This will NOT merge across devices.
    """
    parts = [
        reg.device or "",
        reg.device_series or "",
        reg.peripheral or "",
        reg.register or "",
        # address context
        str(reg.base_address or ""),
        str(reg.address_offset or ""),
        str(reg.full_address or ""),
        # key properties
        str(reg.size or ""),
        str(reg.access or ""),
        str(reg.reset_value or ""),
        # descriptions can be noisy; do NOT use them to decide "exact duplicate"
        # (keep them out of the identity key).
    ]

    # Field structure: strict and ordered
    # Include name + bit range (and description if you want stricter identity)
    for f in sorted(reg.fields, key=lambda x: (x.bit_offset, x.bit_width, x.name or "")):
        parts.append(f"{f.name or ''}:{f.bit_offset}:{f.bit_width}")

    return _hash_str("|".join(parts))


# ----------------------------
# Description dedup (optional)
# ----------------------------

def _compute_description_id(
    peripheral: Optional[str],
    register: Optional[str],
    register_description: Optional[str],
    fields_summary: List[str],
) -> str:
    """
    Canonical description ID used for storage deduplication.

    Important: This is NOT used to merge registers. It only creates a stable key
    for shared description content.

    Include peripheral/register names so "DR" in ADC doesn't collide with "DR" in USART.
    """
    parts = [
        _norm_text(peripheral),
        _norm_text(register),
        _norm_text(register_description),
    ]
    parts.extend(fields_summary)
    return _hash_str("|".join(parts))


def deduplicate_registers_exact(registers: List[ParsedRegister]) -> List[ParsedRegister]:
    """
    Remove only TRUE duplicate register records (usually ingestion duplicates).
    Does not collapse across devices.

    If duplicates exist, keeps the first occurrence.
    """
    seen: set[str] = set()
    out: List[ParsedRegister] = []

    for reg in registers:
        key = _compute_exact_register_key(reg)
        if key in seen:
            continue
        seen.add(key)
        out.append(reg)

    return out


def deduplicate_descriptions(
    registers: List[ParsedRegister],
    *,
    store_field_descriptions: bool = False,
) -> Tuple[List[ParsedRegister], Dict[str, str]]:
    """
    Create a canonical store for repeated description text.

    Returns:
      (registers, description_store)

    - Does NOT remove any register.
    - Does NOT change device/peripheral identity fields.
    - Attaches a "description_id" attribute to ParsedRegister (if your model allows it),
      otherwise it returns the store and you can keep an external mapping.

    description_store maps:
      description_id -> canonical_description_text

    You can later hydrate the full description when building LLM context.
    """
    description_store: Dict[str, str] = {}
    # optional: store field descriptions too
    # (kept simple here; you can extend to a separate store per field)

    for reg in registers:
        # build a stable “fields summary” for the description id
        # (names + bit positions, so the same description but different layout won't collide)
        field_bits = []
        for f in sorted(reg.fields, key=lambda x: (x.bit_offset, x.bit_width, x.name or "")):
            field_bits.append(f"{_norm_text(f.name)}:{f.bit_offset}:{f.bit_width}")

        desc_id = _compute_description_id(
            reg.peripheral,
            reg.register,
            reg.register_description,
            field_bits,
        )

        # canonical description text to store once
        # (you can include peripheral_description too if it’s very repetitive)
        canonical = _norm_text(reg.register_description)

        # Store if new and non-empty
        if canonical and desc_id not in description_store:
            description_store[desc_id] = canonical

        # Attach id to the register if possible.
        # If ParsedRegister is a dataclass without this field, this will raise.
        # In that case, remove this setattr and keep an external mapping.
        try:
            setattr(reg, "description_id", desc_id)
        except Exception:
            pass

        # Optional: field description store (not implemented fully)
        # You can add e.g. field_description_id per field the same way.
        _ = store_field_descriptions

    return registers, description_store


def get_deduplication_stats(
    original: List[ParsedRegister],
    deduplicated: List[ParsedRegister]
) -> Dict[str, int]:
    original_count = len(original)
    deduplicated_count = len(deduplicated)
    reduction_count = original_count - deduplicated_count
    reduction_percent = (reduction_count / original_count * 100) if original_count > 0 else 0

    return {
        "original_count": original_count,
        "deduplicated_count": deduplicated_count,
        "reduction_count": reduction_count,
        "reduction_percent": reduction_percent,
    }
