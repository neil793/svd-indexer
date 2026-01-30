"""
Deduplicate identical registers across chips and peripheral instances
"""
from typing import List, Dict, Tuple
from collections import defaultdict
from .models import ParsedRegister


def _make_dedup_key(register: ParsedRegister) -> Tuple[str, str, str]:
    """
    Create deduplication key from register
    
    Registers are considered identical if they have:
    - Same peripheral family (UART, GPIO, TIM, etc.)
    - Same register name (BRR, DIER, ODR, etc.)
    - Same field structure (names AND bit positions AND widths)
    
    Returns:
        Tuple of (peripheral_family, register_name, field_signature)
    """
    # Use explicit peripheral_group from SVD if available
    # Otherwise fall back to first 4 characters of peripheral name
    
    peripheral_family = register.peripheral_group or register.peripheral[:4]
    
    # Create field signature including bit positions and widths
    # This prevents false matches where field names match but layouts differ
    sorted_fields = sorted(register.fields, key=lambda f: f.name)
    field_parts = []
    for field in sorted_fields:
        # Format: "field_name@bit_offset:bit_width"
        field_parts.append(f"{field.name}@{field.bit_offset}:{field.bit_width}")
    
    field_signature = ",".join(field_parts)
    
    return (peripheral_family, register.register, field_signature)


def deduplicate_registers(registers: List[ParsedRegister]) -> List[ParsedRegister]:
    """
    Deduplicate registers by grouping identical ones
    
    Process:
    1. Group registers by (peripheral_family, register_name, field_structure)
    2. For each group, keep one representative
    3. Track all devices and peripheral instances in metadata
    
    Args:
        registers: List of all parsed registers
        
    Returns:
        List of unique registers with device/instance tracking
    """
    print(f"\nDeduplicating {len(registers)} registers...")
    
    # Group registers by dedup key
    register_groups: Dict[Tuple, List[ParsedRegister]] = defaultdict(list)
    
    for reg in registers:
        key = _make_dedup_key(reg)
        register_groups[key].append(reg)
    
    # Create deduplicated list
    deduplicated = []
    
    for key, group in register_groups.items():
        if len(group) == 1:
            # Unique register - no duplicates found
            deduplicated.append(group[0])
            continue
        
        # Multiple instances found - deduplicate them
        representative = group[0]
        
        # Collect all devices that have this register
        devices = sorted(list(set([r.device for r in group])))
        
        # Collect all peripheral instances that have this register
        peripherals = sorted(list(set([r.peripheral for r in group])))
        
        # Build address map: "device/peripheral" -> address
        address_map = {}
        for reg in group:
            lookup_key = f"{reg.device}/{reg.peripheral}"
            address_map[lookup_key] = reg.full_address
        
        # Attach metadata to representative
        representative.devices = devices
        representative.peripheral_instances = peripherals
        representative.address_map = address_map
        
        # Generalize peripheral name if multiple instances exist
        if len(peripherals) > 1:
            representative.peripheral = representative.peripheral_group or representative.peripheral[:4]
        
        # Generalize device name if multiple devices exist
        if len(devices) > 1:
            representative.device = _get_device_family(group)
        
        deduplicated.append(representative)
    
    # Print summary statistics
    total_before = len(registers)
    total_after = len(deduplicated)
    reduction = ((total_before - total_after) / total_before) * 100
    
    print(f"✓ Deduplication complete:")
    print(f"  Before: {total_before:,} registers")
    print(f"  After:  {total_after:,} unique registers")
    print(f"  Reduction: {reduction:.1f}%")
    
    dedup_counts = [len(group) for group in register_groups.values()]
    max_dupes = max(dedup_counts)
    print(f"  Max duplicates for single register: {max_dupes}")
    
    return deduplicated


def _get_device_family(register_group: List[ParsedRegister]) -> str:
    """
    Generate a generic family name from a group of registers
    
    Uses a three-tier fallback system:
    1. SVD <series> tags (most reliable when present)
    2. Pattern extraction from device names (STM32-specific heuristic)
    3. Device listing or count (fallback for unknown patterns)
    
    Args:
        register_group: List of registers (all functionally identical)
        
    Returns:
        Display string representing the device family
    """
    # Tier 1: Try to use series metadata from SVD files
    series_set = set()
    for reg in register_group:
        if hasattr(reg, 'device_series') and reg.device_series:
            series_set.add(reg.device_series)
    
    if series_set:
        if len(series_set) == 1:
            return list(series_set)[0]
        else:
            return f"Multiple families ({len(series_set)})"
    
    # Tier 2: Try pattern extraction from device names
    devices = sorted(list(set([r.device for r in register_group])))
    extracted_family = _extract_family_from_names(devices)
    
    if extracted_family:
        return extracted_family
    
    # Tier 3: Fall back to listing device names
    if len(devices) == 1:
        return devices[0]
    
    if len(devices) <= 3:
        return ", ".join(devices)
    
    return f"Multiple devices ({len(devices)})"

def _extract_family_from_names(devices: List[str]) -> str | None:
    """
    Attempt to extract family pattern from device names
    
    Works for common ARM vendor naming patterns:
    - STM32: "STM32F407" → "STM32F4"
    - Nordic: "nRF52840" → "nRF52"
    - NXP Kinetis: "MK64FN1M0" → "MK64"
    
    Returns None if no common pattern found.
    """
    if not devices:
        return None
    
    # STM32 pattern: STM32 + Family Letter + Sub-family Digit
    if all(d.startswith("STM32") for d in devices):
        families = set()
        for device in devices:
            suffix = device[5:]  # Remove "STM32" prefix
            if len(suffix) >= 2:
                family = suffix[:2]  # "F407" -> "F4"
                families.add(family)
        
        if len(families) == 1:
            return f"STM32{list(families)[0]}xx"
        elif len(families) > 1:
            return f"STM32 (multiple families)"
    
    # Nordic pattern: nRF + Series Number
    if all(d.startswith("nRF") for d in devices):
        families = set()
        for device in devices:
            if len(device) >= 5:
                family = device[:5]  # "nRF52840" -> "nRF52"
                families.add(family)
        
        if len(families) == 1:
            return f"{list(families)[0]}xxx"
        elif len(families) > 1:
            return "nRF (multiple series)"
    
    # NXP Kinetis pattern: MK + Series Number
    if all(d.startswith("MK") for d in devices):
        families = set()
        for device in devices:
            if len(device) >= 4:
                family = device[:4]  # "MK64FN1M0" -> "MK64"
                families.add(family)
        
        if len(families) == 1:
            return f"{list(families)[0]}xxx"
        elif len(families) > 1:
            return "Kinetis (multiple series)"
    
    # No recognizable pattern
    return None
