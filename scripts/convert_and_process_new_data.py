#!/usr/bin/env python3
"""Copy new OG data, convert to CSV, and process."""

import subprocess
import sys
from pathlib import Path
import shutil

og_dir = Path('data/OG_data')
imu_raw_dir = Path('data/00_collect/imu_raw')

print("="*80)
print("PROCESSING NEW OG DATA")
print("="*80)

# Step 1: Copy files
print("\n1. Copying OG files to imu_raw...")
xlsx_files = list(og_dir.glob('*.xlsx'))
if not xlsx_files:
    print(f"  ✗ No XLSX files found in {og_dir}")
    sys.exit(1)

# Clean imu_raw first
for f in imu_raw_dir.glob('*.csv'):
    f.unlink()
for f in imu_raw_dir.glob('*.xlsx'):
    f.unlink()

# Copy new files
for xlsx_file in xlsx_files:
    shutil.copy2(xlsx_file, imu_raw_dir / xlsx_file.name)
    print(f"  ✓ Copied: {xlsx_file.name}")

print(f"\n  Copied {len(xlsx_files)} files")

# Step 2: Convert to CSV
print("\n2. Converting XLSX to CSV...")
result = subprocess.run(
    [sys.executable, 'scripts/convert_xlsx.py', str(imu_raw_dir)],
    capture_output=True,
    text=True
)
print(result.stdout)
if result.returncode != 0:
    print("Error:", result.stderr)
    sys.exit(1)

print("\n" + "="*80)
print("✅ CONVERSION COMPLETE")
print("="*80)
print("\nReady to run pipeline:")
print("  PYTHONPATH=src python scripts/pipeline_with_real_data.py")
