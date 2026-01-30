"""
Data models for SVD indexing pipeline
"""
from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class ParsedField:
    """
    Represents a bit field within a register
    """
    name: str
    description: Optional[str]
    bit_offset: int
    bit_width: int
    bit_range: str  # e.g., "[15:4]"
    access: Optional[str]  # "read-write", "read-only", "write-only"


@dataclass
class ParsedRegister:
    """
    Complete register information extracted from SVD
    """
    # Identity
    device: str
    device_series: str | None
    peripheral: str
    peripheral_description: Optional[str]
    peripheral_group: Optional[str]  # Group name (e.g., "USART")
    register: str
    register_description: Optional[str]
    
    # Address information
    base_address: str  # Peripheral base address
    address_offset: str  # Register offset from base
    full_address: str  # Combined address
    
    # Register properties
    size: int  # Size in bits (8, 16, 32)
    access: Optional[str]  # Access permission
    reset_value: Optional[str]  # Reset value (hex)
    
    # Field information
    fields: List[ParsedField]

    #Deduplication tracking
    devices: Optional[List[str]] = None             
    peripheral_instances: Optional[List[str]] = None 
    address_map: Optional[Dict[str, str]] = None 


@dataclass
class TextChunk:
    """
    Searchable text chunk ready for embedding
    """
    id: str  # Unique ID: "device/peripheral/register"
    text: str  # Searchable text content
    metadata: dict  # Metadata for retrieval and filtering