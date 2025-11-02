#!/usr/bin/env python3
"""Convert XLSX files to CSV for the IMU pipeline."""

import pandas as pd
import sys
from pathlib import Path

# Get target directory from command line or use default
if len(sys.argv) > 1:
    this_dir = Path(sys.argv[1])
else:
    this_dir = Path(__file__).parent.parent / 'data' / '00_collect' / 'imu_raw'

print("="*80)
print("CONVERTING XLSX FILES TO CSV")
print("="*80)

xlsx_files = list(this_dir.glob('*.xlsx'))
csv_files = list(this_dir.glob('*.csv'))

print(f"\nFound:")
print(f"  XLSX files: {len(xlsx_files)}")
print(f"  CSV files: {len(csv_files)}")

if not xlsx_files:
    print("\n✓ No XLSX files to convert")
    exit(0)

converted = 0
for xlsx_file in xlsx_files:
    print(f"\nConverting: {xlsx_file.name}")
    
    try:
        # Use openpyxl directly to avoid formula parsing issues
        import openpyxl
        wb = openpyxl.load_workbook(xlsx_file, data_only=True)
        ws = wb.active
        
        # Read data manually
        data = []
        headers = []
        
        for idx, row in enumerate(ws.iter_rows(values_only=True)):
            if idx == 0:
                headers = list(row)
            else:
                data.append(row)
        
        df = pd.DataFrame(data, columns=headers)
        
        csv_file = xlsx_file.with_suffix('.csv')
        df.to_csv(csv_file, index=False)
        converted += 1
        print(f"  ✓ Saved: {csv_file.name} ({len(df)} rows)")
        
        # Optionally delete XLSX
        # xlsx_file.unlink()
        # print(f"  ✓ Deleted original XLSX")
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")

print("\n" + "="*80)
print(f"✓ Converted {converted} files")
print("="*80)

