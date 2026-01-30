"""
Create searchable text chunks from parsed SVD registers
Improved for semantic retrieval + grounded LLM responses
"""
from typing import List, Dict, Any
from collections import defaultdict
import hashlib
from .models import ParsedRegister, TextChunk
from .config import config


def create_chunk(register: ParsedRegister) -> TextChunk:
    """
    One chunk per register with all field details.
    Text is formatted to be embedding-friendly and to support grounded answers.
    Includes deduplication metadata when available.
    """
    lines: List[str] = []

    # Header context
    if config.include_device:
        lines.append(f"Device: {register.device}")
        
        # Show which devices this applies to (if deduplicated)
        if hasattr(register, 'devices') and register.devices and len(register.devices) > 1:
            device_list = ', '.join(register.devices[:5])
            if len(register.devices) > 5:
                device_list += f" (and {len(register.devices) - 5} more)"
            lines.append(f"Applies to devices: {device_list}")

    # Peripheral line
    peripheral_line = f"Peripheral: {register.peripheral}"
    if config.include_peripheral_desc and register.peripheral_description:
        peripheral_line += f" — {register.peripheral_description}"
    if register.peripheral_group:
        peripheral_line += f" (Group: {register.peripheral_group})"
    lines.append(peripheral_line)
    
    # Show which peripheral instances (if deduplicated)
    if hasattr(register, 'peripheral_instances') and register.peripheral_instances:
        if len(register.peripheral_instances) > 1:
            periph_list = ', '.join(register.peripheral_instances[:8])
            if len(register.peripheral_instances) > 8:
                periph_list += f" (and {len(register.peripheral_instances) - 8} more)"
            lines.append(f"Peripheral instances: {periph_list}")

    # Register line
    reg_line = f"Register: {register.register}"
    if config.include_register_desc and register.register_description:
        reg_line += f" — {register.register_description}"
    lines.append(reg_line)

    # Address handling - show examples if deduplicated, otherwise show single address
    if hasattr(register, 'address_map') and register.address_map and len(register.address_map) > 1:
        lines.append("Example addresses:")
        for key, addr in list(register.address_map.items())[:3]:
            lines.append(f"  {key}: {addr}")
        if len(register.address_map) > 3:
            lines.append(f"  (and {len(register.address_map) - 3} more - see metadata)")
    else:
        lines.append(f"Address: {register.full_address}")
    
    # Size and access
    lines.append(f"Size: {register.size} bits")
    if register.access:
        lines.append(f"Access: {register.access}")
    if register.reset_value:
        lines.append(f"Reset value: {register.reset_value}")

    # Fields
    if config.include_field_names and register.fields:
        lines.append("Fields:")
        for field in register.fields:
            desc = f" — {field.description}" if (config.include_field_desc and field.description) else ""
            access_str = f" ({field.access})" if field.access else ""
            lines.append(f"- {field.name} {field.bit_range}{access_str}{desc}")

    text = "\n".join(lines)

    # Enhanced metadata with deduplication info
    metadata: Dict[str, Any] = {
        "type": "register",
        "device": register.device,
        "peripheral": register.peripheral,
        "peripheral_group": register.peripheral_group or "",
        "register": register.register,
        "address": register.full_address,
        "size": int(register.size) if register.size is not None else 0,
        "access": register.access or "",
        "field_names": ", ".join([f.name for f in register.fields]) if register.fields else "",
        
        # Lexical search helpers
        "peripheral_lower": (register.peripheral or "").lower(),
        "register_lower": (register.register or "").lower(),
        
        # Deduplication metadata (critical for looking up specific addresses)
        "devices": getattr(register, 'devices', [register.device]),
        "peripheral_instances": getattr(register, 'peripheral_instances', [register.peripheral]),
        "address_map": getattr(register, 'address_map', {
            f"{register.device}/{register.peripheral}": register.full_address
        }),
    }

    # Generate stable chunk ID using hash of dedup key components
    # This prevents duplicate IDs when registers are deduplicated
    id_components = [
        register.peripheral_group or register.peripheral,
        register.register,
        ",".join(sorted([f"{f.name}@{f.bit_offset}:{f.bit_width}" for f in register.fields]))
    ]
    id_string = "|".join(id_components)
    chunk_hash = hashlib.md5(id_string.encode()).hexdigest()[:12]
    
    chunk_id = f"{register.peripheral}_{register.register}_{chunk_hash}"

    return TextChunk(id=chunk_id, text=text, metadata=metadata)


def create_device_summary_chunks(registers: List[ParsedRegister]) -> List[TextChunk]:
    """
    Create summary chunks for each device listing available peripherals
    
    Enables queries like:
    - "What timers are available on STM32F030?"
    - "Does STM32F407 have USB?"
    - "List all peripherals on device X"
    
    Args:
        registers: List of parsed registers (possibly deduplicated)
        
    Returns:
        List of device summary chunks (one per device)
    """
    # Group peripherals by device
    device_peripherals = defaultdict(set)
    device_series = {}
    
    for reg in registers:
        # Get actual device names (handle deduplication)
        if hasattr(reg, 'devices') and reg.devices:
            devices = reg.devices
        else:
            devices = [reg.device]
        
        # Get peripheral instances (handle deduplication)
        if hasattr(reg, 'peripheral_instances') and reg.peripheral_instances:
            periphs = reg.peripheral_instances
        else:
            periphs = [reg.peripheral]
        
        # Track which device has which peripherals
        for device in devices:
            for periph in periphs:
                device_peripherals[device].add(periph)
            
            # Store series info if available
            if hasattr(reg, 'device_series') and reg.device_series:
                device_series[device] = reg.device_series
    
    # Create one summary chunk per device
    summary_chunks = []
    
    for device, peripherals in device_peripherals.items():
        # Group peripherals by type for better organization
        timers = sorted([p for p in peripherals if p.startswith("TIM")])
        gpios = sorted([p for p in peripherals if p.startswith("GPIO")])
        uarts = sorted([p for p in peripherals if "UART" in p or "USART" in p])
        spis = sorted([p for p in peripherals if p.startswith("SPI")])
        i2cs = sorted([p for p in peripherals if p.startswith("I2C")])
        adcs = sorted([p for p in peripherals if p.startswith("ADC")])
        dacs = sorted([p for p in peripherals if p.startswith("DAC")])
        dmas = sorted([p for p in peripherals if p.startswith("DMA")])
        usbs = sorted([p for p in peripherals if "USB" in p])
        cans = sorted([p for p in peripherals if "CAN" in p])
        
        # Collect "other" peripherals (system/misc)
        categorized = set(timers + gpios + uarts + spis + i2cs + adcs + dacs + dmas + usbs + cans)
        others = sorted([p for p in peripherals if p not in categorized])
        
        # Build text
        lines = [f"Device: {device}"]
        
        if device in device_series:
            lines.append(f"Series: {device_series[device]}")
        
        lines.append(f"\nAvailable Peripherals ({len(peripherals)} total):\n")
        
        # List each peripheral type
        if timers:
            lines.append(f"Timers ({len(timers)}): {', '.join(timers)}")
        if gpios:
            lines.append(f"GPIO ({len(gpios)}): {', '.join(gpios)}")
        if uarts:
            lines.append(f"UART/USART ({len(uarts)}): {', '.join(uarts)}")
        if spis:
            lines.append(f"SPI ({len(spis)}): {', '.join(spis)}")
        if i2cs:
            lines.append(f"I2C ({len(i2cs)}): {', '.join(i2cs)}")
        if adcs:
            lines.append(f"ADC ({len(adcs)}): {', '.join(adcs)}")
        if dacs:
            lines.append(f"DAC ({len(dacs)}): {', '.join(dacs)}")
        if dmas:
            lines.append(f"DMA ({len(dmas)}): {', '.join(dmas)}")
        if usbs:
            lines.append(f"USB ({len(usbs)}): {', '.join(usbs)}")
        if cans:
            lines.append(f"CAN ({len(cans)}): {', '.join(cans)}")
        
        # List other peripherals at the end
        if others:
            lines.append(f"\nSystem/Other: {', '.join(others)}")
        
        text = "\n".join(lines)
        
        # Create chunk with metadata
        chunk = TextChunk(
            id=f"device_summary_{device}",
            text=text,
            metadata={
                "type": "device_summary",
                "device": device,
                "device_series": device_series.get(device, ""),
                "peripheral_count": len(peripherals),
                "peripherals": sorted(list(peripherals)),
                "timers": timers,
                "gpios": gpios,
                "uarts": uarts,
                "spis": spis,
                "i2cs": i2cs,
                "adcs": adcs,
                "dacs": dacs,
                "usbs": usbs,
                "cans": cans,
            }
        )
        
        summary_chunks.append(chunk)
    
    return summary_chunks


def create_chunks(registers: List[ParsedRegister]) -> List[TextChunk]:
    """
    Create text chunks from all registers
    
    Creates two types of chunks:
    1. Register-level chunks (one per unique register)
    2. Device summary chunks (one per device)
    
    Args:
        registers: List of parsed (and possibly deduplicated) registers
        
    Returns:
        List of all text chunks ready for embedding
    """
    chunks: List[TextChunk] = []
    chunk_id_set = set()
    duplicate_count = 0

    # Create register-level chunks
    for register in registers:
        try:
            chunk = create_chunk(register)

            # Check for duplicate IDs (shouldn't happen with hash-based IDs, but just in case)
            if chunk.id in chunk_id_set:
                duplicate_count += 1
                # Make unique by appending counter
                chunk.id = f"{chunk.id}_dup{duplicate_count}"
                print(f"⚠️  Duplicate chunk ID detected: {chunk.id}")
            else:
                chunk_id_set.add(chunk.id)

            chunks.append(chunk)

        except Exception as e:
            device_id = getattr(register, 'device', 'unknown')
            periph_id = getattr(register, 'peripheral', 'unknown')
            reg_id = getattr(register, 'register', 'unknown')
            print(f"⚠️  Failed to create chunk for {device_id}/{periph_id}/{reg_id}: {e}")

    # Report register chunk status
    print(f"\n✓ Created {len(chunks)} register-level chunks")
    if duplicate_count > 0:
        print(f"⚠️  Fixed {duplicate_count} duplicate chunk IDs")
    
    # Create device summary chunks
    print("\nCreating device summary chunks...")
    summary_chunks = create_device_summary_chunks(registers)
    chunks.extend(summary_chunks)
    print(f"✓ Created {len(summary_chunks)} device summary chunks")
    
    # Final summary
    print(f"\n✓ Total: {len(chunks)} chunks ({len(chunks) - len(summary_chunks)} register-level + {len(summary_chunks)} device summaries)")
    
    return chunks