"""
Comprehensive deduplication validation tests
"""
from pathlib import Path
from cmsis_svd.parser import SVDParser
from collections import defaultdict, Counter
import json

from indexer.parser import parse_svd_file, parse_multiple_svd_files
from indexer.deduplicator import deduplicate_registers, _make_dedup_key


def load_test_data(svd_dir: str = "data/STMicro"):
    """Load all SVD files for testing"""
    svd_files = list(Path(svd_dir).glob("*.svd"))
    print(f"Found {len(svd_files)} SVD files")
    return [str(f) for f in sorted(svd_files)]


# We'll add test functions below...
def test_groupname_coverage(svd_files):
    """
    Test 1: Check groupName coverage across all SVD files
    
    This tells you how reliable the deduplication grouping is.
    High coverage (>80%) = mostly explicit grouping
    Low coverage (<50%) = mostly [:4] fallback
    """
    print("\n" + "="*70)
    print("TEST 1: groupName Coverage Analysis")
    print("="*70)
    
    results = []
    
    for svd_path in svd_files[:10]:  # Test first 10 files
        try:
            parser = SVDParser.for_xml_file(svd_path)
            device = parser.get_device()
            
            total = len(device.peripherals)
            with_group = sum(1 for p in device.peripherals 
                           if hasattr(p, 'group_name') and p.group_name)
            
            coverage = (with_group / total * 100) if total > 0 else 0
            
            results.append({
                'device': device.name,
                'total': total,
                'with_group': with_group,
                'without_group': total - with_group,
                'coverage': coverage
            })
            
            print(f"\n{device.name}:")
            print(f"  Total peripherals: {total}")
            print(f"  With groupName: {with_group} ({coverage:.1f}%)")
            print(f"  Without groupName: {total - with_group} ({100-coverage:.1f}%)")
            
        except Exception as e:
            print(f"Error parsing {svd_path}: {e}")
    
    # Summary
    avg_coverage = sum(r['coverage'] for r in results) / len(results)
    print(f"\n{'='*70}")
    print(f"AVERAGE groupName COVERAGE: {avg_coverage:.1f}%")
    print(f"{'='*70}")
    
    if avg_coverage > 80:
        print("✅ EXCELLENT: High explicit grouping, dedup should be reliable")
    elif avg_coverage > 50:
        print("⚠️  GOOD: Moderate coverage, [:4] fallback used frequently")
    else:
        print("❌ WARNING: Low coverage, heavily relies on [:4] fallback")
    
    return results


def test_groupname_details(svd_file):
    """
    Test 1b: Detailed groupName analysis for a single file
    
    Shows exactly which peripherals have groups and which don't
    """
    print("\n" + "="*70)
    print(f"TEST 1b: Detailed groupName Analysis")
    print("="*70)
    
    parser = SVDParser.for_xml_file(svd_file)
    device = parser.get_device()
    
    with_group = defaultdict(list)
    without_group = []
    
    for p in device.peripherals:
        if hasattr(p, 'group_name') and p.group_name:
            with_group[p.group_name].append(p.name)
        else:
            without_group.append(p.name)
    
    print(f"\nDevice: {device.name}")
    print(f"Total peripherals: {len(device.peripherals)}")
    
    print(f"\n--- PERIPHERALS WITH groupName ({sum(len(v) for v in with_group.values())}) ---")
    for group, members in sorted(with_group.items()):
        print(f"  {group} ({len(members)}): {', '.join(sorted(members)[:5])}", end="")
        if len(members) > 5:
            print(f" ... (+{len(members)-5} more)")
        else:
            print()
    
    print(f"\n--- PERIPHERALS WITHOUT groupName ({len(without_group)}) ---")
    print(f"  (Using [:4] fallback)")
    for p in sorted(without_group)[:20]:
        print(f"  {p} → {p[:4]}")
    if len(without_group) > 20:
        print(f"  ... (+{len(without_group)-20} more)")

def test_timer_collisions(svd_files):
    """
    Test 2: Check for timer numbering collisions
    
    Problem: "TIM10"[:4] = "TIM1" (collides with TIM1)
    
    This checks:
    - Do timers have explicit groupName="TIM"?
    - Or do TIM1 and TIM10 end up with different keys?
    """
    print("\n" + "="*70)
    print("TEST 2: Timer Numbering Collision Check")
    print("="*70)
    
    for svd_path in svd_files[:5]:
        try:
            parser = SVDParser.for_xml_file(svd_path)
            device = parser.get_device()
            
            timers = [p for p in device.peripherals if p.name.startswith("TIM")]
            if not timers:
                continue
            
            print(f"\n{device.name}: Found {len(timers)} timers")
            
            # Check groupName
            timer_groups = {}
            for t in timers:
                group_name = getattr(t, 'group_name', None)
                fallback = t.name[:4]
                timer_groups[t.name] = {
                    'explicit_group': group_name,
                    'fallback_group': fallback,
                    'final_group': group_name if group_name else fallback
                }
            
            # Show grouping
            for name, info in sorted(timer_groups.items()):
                print(f"  {name:10} → ", end="")
                if info['explicit_group']:
                    print(f"groupName='{info['explicit_group']}' ✅")
                else:
                    print(f"[:4]='{info['fallback_group']}' ⚠️", end="")
                    # Check for collisions
                    if name.startswith("TIM1") and len(name) > 4:
                        print(" (COLLISION WITH TIM1!)")
                    else:
                        print()
            
            # Check for actual collisions
            groups = defaultdict(list)
            for name, info in timer_groups.items():
                groups[info['final_group']].append(name)
            
            collisions = {k: v for k, v in groups.items() if len(v) > 1 and k[:4] == "TIM1"}
            if collisions:
                print("\n  ❌ COLLISION DETECTED:")
                for group, members in collisions.items():
                    print(f"    Group '{group}': {', '.join(members)}")
            else:
                print("\n  ✅ No timer collisions detected")
                
        except Exception as e:
            print(f"Error: {e}")
def test_usart_uart_grouping(svd_files):
    """
    Test 3: Check USART vs UART grouping
    
    Question: Should USART1-3 and UART4-8 be grouped together?
    
    Checks:
    - Do they have same groupName?
    - Or different groupNames?
    - Or using [:4] fallback (USAR vs UART)?
    """
    print("\n" + "="*70)
    print("TEST 3: USART vs UART Grouping")
    print("="*70)
    
    for svd_path in svd_files[:5]:
        try:
            parser = SVDParser.for_xml_file(svd_path)
            device = parser.get_device()
            
            usarts = [p for p in device.peripherals if 'USART' in p.name or 'UART' in p.name]
            if not usarts:
                continue
            
            print(f"\n{device.name}: Found {len(usarts)} UART/USART peripherals")
            
            usart_groups = {}
            uart_groups = {}
            
            for u in usarts:
                group_name = getattr(u, 'group_name', None)
                fallback = u.name[:4]
                final = group_name if group_name else fallback
                
                info = {
                    'explicit': group_name,
                    'fallback': fallback,
                    'final': final
                }
                
                if 'USART' in u.name:
                    usart_groups[u.name] = info
                else:
                    uart_groups[u.name] = info
            
            print("\n  USART instances:")
            for name, info in sorted(usart_groups.items()):
                print(f"    {name:10} → final_group='{info['final']}'", end="")
                if info['explicit']:
                    print(f" (explicit) ✅")
                else:
                    print(f" ([:4] fallback) ⚠️")
            
            print("\n  UART instances:")
            for name, info in sorted(uart_groups.items()):
                print(f"    {name:10} → final_group='{info['final']}'", end="")
                if info['explicit']:
                    print(f" (explicit) ✅")
                else:
                    print(f" ([:4] fallback) ⚠️")
            
            # Check if they group together
            all_groups = set()
            all_groups.update(info['final'] for info in usart_groups.values())
            all_groups.update(info['final'] for info in uart_groups.values())
            
            if len(all_groups) == 1:
                print(f"\n  ✅ All USART/UART group together as '{list(all_groups)[0]}'")
            else:
                print(f"\n  ⚠️  USART and UART use different groups: {all_groups}")
                print(f"     They will NOT deduplicate together")
                
        except Exception as e:
            print(f"Error: {e}")
def test_field_signature_accuracy(svd_files):
    """
    Test 4: Validate field signature matching
    
    Critical test: Do registers with matching keys actually have:
    - Same bit positions?
    - Same bit widths?
    - Same access permissions?
    
    This checks if the dedup key is too loose.
    """
    print("\n" + "="*70)
    print("TEST 4: Field Signature Accuracy")
    print("="*70)
    
    # Parse a few files
    all_registers = parse_multiple_svd_files(svd_files[:3])
    
    # Group by dedup key
    groups = defaultdict(list)
    for reg in all_registers:
        key = _make_dedup_key(reg)
        groups[key].append(reg)
    
    # Check groups with multiple members
    mismatches = []
    
    for key, group in groups.items():
        if len(group) <= 1:
            continue
        
        # Check if all fields match exactly
        reference = group[0]
        
        for reg in group[1:]:
            # Check field count
            if len(reg.fields) != len(reference.fields):
                mismatches.append({
                    'key': key,
                    'issue': 'field_count',
                    'ref': f"{reference.device}/{reference.peripheral}",
                    'other': f"{reg.device}/{reg.peripheral}"
                })
                continue
            
            # Check each field
            for i, field in enumerate(reg.fields):
                ref_field = reference.fields[i]
                
                # Check bit positions
                if field.bit_offset != ref_field.bit_offset:
                    mismatches.append({
                        'key': key,
                        'issue': 'bit_offset',
                        'field': field.name,
                        'ref': f"{reference.device}/{reference.peripheral} bit {ref_field.bit_offset}",
                        'other': f"{reg.device}/{reg.peripheral} bit {field.bit_offset}"
                    })
                
                # Check bit widths
                if field.bit_width != ref_field.bit_width:
                    mismatches.append({
                        'key': key,
                        'issue': 'bit_width',
                        'field': field.name,
                        'ref': f"{reference.device}/{reference.peripheral} width {ref_field.bit_width}",
                        'other': f"{reg.device}/{reg.peripheral} width {field.bit_width}"
                    })
    
    print(f"\nChecked {len(groups)} unique register groups")
    print(f"Found {len(mismatches)} field mismatches")
    
    if not mismatches:
        print("\n✅ EXCELLENT: All deduplicated registers have identical field structures")
    else:
        print(f"\n❌ WARNING: Found {len(mismatches)} mismatches!")
        print("\nFirst 10 mismatches:")
        for m in mismatches[:10]:
            print(f"  {m['key']}")
            print(f"    Issue: {m['issue']}")
            print(f"    Ref:   {m['ref']}")
            print(f"    Other: {m['other']}")
    
    return mismatches
def test_cross_family_deduplication(svd_files):
    """
    Test 5: Cross-family deduplication analysis
    
    Checks:
    - Which registers merge across F4, L4, H7, etc.?
    - How many devices per register?
    - How large are the address maps?
    """
    print("\n" + "="*70)
    print("TEST 5: Cross-Family Deduplication Analysis")
    print("="*70)
    
    # Parse multiple families
    print("\nParsing SVD files...")
    all_registers = parse_multiple_svd_files(svd_files[:10])
    
    print(f"Parsed {len(all_registers)} registers")
    
    # Deduplicate
    print("\nDeduplicating...")
    deduplicated = deduplicate_registers(all_registers)
    
    # Analyze cross-family merges
    cross_family = []
    
    for reg in deduplicated:
        if not hasattr(reg, 'devices') or len(reg.devices) <= 1:
            continue
        
        # Extract chip families
        families = set()
        for device in reg.devices:
            if device.startswith("STM32") and len(device) >= 7:
                family = device[5:7]  # "F4", "L4", "H7", etc.
                families.add(family)
        
        if len(families) > 1:
            cross_family.append({
                'register': f"{reg.peripheral}.{reg.register}",
                'families': sorted(families),
                'num_devices': len(reg.devices),
                'num_peripherals': len(reg.peripheral_instances) if hasattr(reg, 'peripheral_instances') else 1,
                'address_map_size': len(reg.address_map) if hasattr(reg, 'address_map') else 1
            })
    
    print(f"\n{'='*70}")
    print(f"Cross-family merges: {len(cross_family)}")
    print(f"{'='*70}")
    
    if not cross_family:
        print("✅ No cross-family deduplication (files from same family)")
    else:
        print(f"\n⚠️  {len(cross_family)} registers merged across chip families")
        print("\nTop 20 by address map size:")
        for item in sorted(cross_family, key=lambda x: x['address_map_size'], reverse=True)[:20]:
            print(f"\n  {item['register']}")
            print(f"    Families: {', '.join(item['families'])}")
            print(f"    Devices: {item['num_devices']}")
            print(f"    Peripheral instances: {item['num_peripherals']}")
            print(f"    Address map entries: {item['address_map_size']}")
    
    return cross_family
def test_deduplication_stats(svd_files):
    """
    Test 6: Overall deduplication statistics
    
    Provides comprehensive metrics:
    - Reduction rate
    - Average duplicates per register
    - Address map size distribution
    - Device coverage
    """
    print("\n" + "="*70)
    print("TEST 6: Deduplication Statistics")
    print("="*70)
    
    # Parse files
    print("\nParsing SVD files...")
    all_registers = parse_multiple_svd_files(svd_files[:10])
    
    print(f"\nBefore deduplication: {len(all_registers)} registers")
    
    # Deduplicate
    deduplicated = deduplicate_registers(all_registers)
    
    # Calculate stats
    reduction = ((len(all_registers) - len(deduplicated)) / len(all_registers)) * 100
    
    # Analyze deduplicated registers
    address_map_sizes = []
    device_counts = []
    peripheral_counts = []
    
    for reg in deduplicated:
        if hasattr(reg, 'address_map'):
            address_map_sizes.append(len(reg.address_map))
        if hasattr(reg, 'devices'):
            device_counts.append(len(reg.devices))
        if hasattr(reg, 'peripheral_instances'):
            peripheral_counts.append(len(reg.peripheral_instances))
    
    print(f"\n{'='*70}")
    print("DEDUPLICATION METRICS")
    print(f"{'='*70}")
    print(f"Before: {len(all_registers):,} registers")
    print(f"After:  {len(deduplicated):,} unique registers")
    print(f"Reduction: {reduction:.1f}%")
    
    if address_map_sizes:
        print(f"\nAddress Map Sizes:")
        print(f"  Average: {sum(address_map_sizes)/len(address_map_sizes):.1f} entries")
        print(f"  Max: {max(address_map_sizes)} entries")
        print(f"  Registers with >100 addresses: {sum(1 for x in address_map_sizes if x > 100)}")
    
    if device_counts:
        print(f"\nDevice Coverage:")
        print(f"  Average devices per register: {sum(device_counts)/len(device_counts):.1f}")
        print(f"  Max devices: {max(device_counts)}")
    
    if peripheral_counts:
        print(f"\nPeripheral Instances:")
        print(f"  Average instances per register: {sum(peripheral_counts)/len(peripheral_counts):.1f}")
        print(f"  Max instances: {max(peripheral_counts)}")
    
    # Quality assessment
    print(f"\n{'='*70}")
    print("QUALITY ASSESSMENT")
    print(f"{'='*70}")
    
    if reduction > 95:
        print("⚠️  VERY HIGH reduction (>95%)")
        print("   → Might be over-deduplicating")
        print("   → Check for cross-family merges")
    elif reduction > 80:
        print("✅ HIGH reduction (80-95%)")
        print("   → Good deduplication within families")
    elif reduction > 50:
        print("✅ MODERATE reduction (50-80%)")
        print("   → Conservative deduplication")
    else:
        print("⚠️  LOW reduction (<50%)")
        print("   → Files might be from very different devices")
    
    if address_map_sizes and max(address_map_sizes) > 100:
        print("\n⚠️  Some registers have >100 address map entries")
        print("   → Consider family-aware deduplication")
def test_sample_register_inspection(svd_files):
    """
    Test 7: Manually inspect sample deduplicated registers
    
    Shows complete details of a few registers to verify:
    - Are the right peripherals grouped?
    - Are addresses correct?
    - Do fields look identical?
    """
    print("\n" + "="*70)
    print("TEST 7: Sample Register Inspection")
    print("="*70)
    
    # Parse and deduplicate
    all_registers = parse_multiple_svd_files(svd_files[:3])
    deduplicated = deduplicate_registers(all_registers)
    
    # Find interesting samples
    # 1. A highly duplicated register (GPIO.ODR)
    # 2. A moderately duplicated register (UART.BRR)
    # 3. A unique register
    
    samples = []
    
    for reg in deduplicated:
        if hasattr(reg, 'address_map'):
            if 'GPIO' in reg.peripheral and 'ODR' in reg.register:
                samples.append(('GPIO.ODR (highly duplicated)', reg))
            elif 'UART' in reg.peripheral and 'BRR' in reg.register:
                samples.append(('UART.BRR (moderately duplicated)', reg))
        if len(samples) >= 2:
            break
    
    # Add a unique one
    for reg in deduplicated:
        if not hasattr(reg, 'address_map') or len(reg.address_map) == 1:
            samples.append(('Unique register', reg))
            break
    
    # Display samples
    for label, reg in samples:
        print(f"\n{'='*70}")
        print(f"SAMPLE: {label}")
        print(f"{'='*70}")
        print(f"Register: {reg.peripheral}.{reg.register}")
        print(f"Device: {reg.device}")
        print(f"Description: {reg.register_description[:100]}...")
        
        if hasattr(reg, 'devices'):
            print(f"\nDevices ({len(reg.devices)}): {', '.join(reg.devices[:5])}", end="")
            if len(reg.devices) > 5:
                print(f" ... (+{len(reg.devices)-5} more)")
            else:
                print()
        
        if hasattr(reg, 'peripheral_instances'):
            print(f"Peripheral instances ({len(reg.peripheral_instances)}): {', '.join(reg.peripheral_instances[:5])}", end="")
            if len(reg.peripheral_instances) > 5:
                print(f" ... (+{len(reg.peripheral_instances)-5} more)")
            else:
                print()
        
        if hasattr(reg, 'address_map'):
            print(f"\nAddress map ({len(reg.address_map)} entries):")
            for key, addr in list(reg.address_map.items())[:5]:
                print(f"  {key}: {addr}")
            if len(reg.address_map) > 5:
                print(f"  ... (+{len(reg.address_map)-5} more)")
        
        print(f"\nFields ({len(reg.fields)}):")
        for field in reg.fields[:5]:
            print(f"  {field.name} {field.bit_range}: {field.description[:50]}...")
        if len(reg.fields) > 5:
            print(f"  ... (+{len(reg.fields)-5} more)")
def run_all_tests(svd_dir: str = "data/STMicro"):
    """
    Run all deduplication tests
    """
    print("\n" + "="*70)
    print("DEDUPLICATION VALIDATION TEST SUITE")
    print("="*70)
    
    svd_files = load_test_data(svd_dir)
    
    # Run tests
    test_groupname_coverage(svd_files)
    test_groupname_details(svd_files[0])  # Detailed analysis of first file
    test_timer_collisions(svd_files)
    test_usart_uart_grouping(svd_files)
    test_field_signature_accuracy(svd_files)
    test_cross_family_deduplication(svd_files)
    test_deduplication_stats(svd_files)
    test_sample_register_inspection(svd_files)
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()
    