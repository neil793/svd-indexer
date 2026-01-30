"""
Parse SVD files into structured Python objects
"""
from cmsis_svd.parser import SVDParser
from typing import List
from .models import ParsedRegister, ParsedField


def parse_svd_file(
    svd_path: str,
    skip_reserved: bool = True,
    skip_no_description: bool = True
) -> List[ParsedRegister]:
    """
    Parse an SVD file and extract all register information
    
    Args:
        svd_path: Path to .svd XML file
        skip_reserved: Skip fields named "reserved"
        skip_no_description: Skip registers with no description
        
    Returns:
        List of ParsedRegister objects
    """
    parser = SVDParser.for_xml_file(svd_path)
    device = parser.get_device()
    device_series = getattr(device, 'series', None)
    registers = []
    
    for peripheral in device.peripherals:
        # Skip peripherals with no registers
        if not peripheral.registers:
            continue
        
        # Extract peripheral group name if available
        group_name = None
        if hasattr(peripheral, 'group_name') and peripheral.group_name:
            group_name = peripheral.group_name
        
        for register in peripheral.registers:
            # Skip undocumented registers
            # FIX: Use getattr to handle missing description attribute
            register_desc = getattr(register, 'description', None)
            if skip_no_description:
                if not register_desc or not register_desc.strip():
                    continue
            
            # Extract non-reserved fields
            parsed_fields = []
            if register.fields:
                for field in register.fields:
                    # Skip reserved fields (per SVD spec)
                    if skip_reserved and field.name.lower() == "reserved":
                        continue
                    
                    # Calculate bit range
                    bit_range = f"[{field.bit_offset}:{field.bit_offset + field.bit_width - 1}]"
                    
                    parsed_field = ParsedField(
                        name=field.name,
                        description=field.description,
                        bit_offset=field.bit_offset,
                        bit_width=field.bit_width,
                        bit_range=bit_range,
                        access=str(field.access) if field.access else None
                    )
                    parsed_fields.append(parsed_field)
            
            # Calculate full address
            full_address = peripheral.base_address + register.address_offset
            
            # Create ParsedRegister object
            # FIX: Use getattr for peripheral.description
            parsed = ParsedRegister(
                # Identity
                device=device.name,
                device_series=device_series,
                peripheral=peripheral.name,
                peripheral_description=getattr(peripheral, 'description', None) or "",
                peripheral_group=group_name,
                register=register.name,
                register_description=register_desc or "",
                
                # Address
                base_address=hex(peripheral.base_address),
                address_offset=hex(register.address_offset),
                full_address=hex(full_address),
                
                # Properties
                size=register.size if register.size else 32,
                access=str(register.access) if register.access else None,
                reset_value=hex(register.reset_value) if register.reset_value is not None else None,
                
                # Fields
                fields=parsed_fields,
            )
            
            registers.append(parsed)
    
    return registers


def parse_multiple_svd_files(svd_paths: List[str]) -> List[ParsedRegister]:
    """
    Parse multiple SVD files
    
    Args:
        svd_paths: List of paths to .svd files
        
    Returns:
        Combined list of all ParsedRegister objects
    """
    all_registers = []
    
    for svd_path in svd_paths:
        try:
            registers = parse_svd_file(svd_path)
            all_registers.extend(registers)
            print(f"✓ Parsed {svd_path}: {len(registers)} registers")
        except Exception as e:
            print(f"✗ Error parsing {svd_path}: {e}")
    
    return all_registers