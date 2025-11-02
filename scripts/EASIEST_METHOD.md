# Easiest Method to Download SharePoint Files

## Steps

1. **Open SharePoint folder in browser**
   - Click the link you provided
   
2. **Get direct download URLs**
   - Right-click each file → "Copy link"
   - OR Use "Download" to get temporary links
   
3. **Create a CSV file** (e.g., `files_to_download.csv`):

```
url,filename
https://ndusbpos.sharepoint.com/.../file1.xlsx?download=1,file1.xlsx
https://ndusbpos.sharepoint.com/.../file2.xlsx?download=1,file2.xlsx
```

**OR** if the SharePoint links have `?e=abc123`, add `&download=1`:

```
url,filename
https://ndusbpos.sharepoint.com/.../file1.xlsx?e=abc123&download=1,file1.xlsx
```

4. **Run the script**:

```bash
python scripts/download_sharepoint_data.py --csv-urls files_to_download.csv
```

Files will be:
- Downloaded to `data/00_collect/imu_raw/`
- Automatically converted from XLSX to CSV

---

## Alternative: Manual Download

If the script doesn't work, just:
1. Download files manually from SharePoint
2. Save XLSX files to `data/00_collect/imu_raw/`
3. Convert to CSV:

```python
import pandas as pd
from pathlib import Path

for xlsx_file in Path('data/00_collect/imu_raw').glob('*.xlsx'):
    df = pd.read_excel(xlsx_file)
    csv_file = xlsx_file.with_suffix('.csv')
    df.to_csv(csv_file, index=False)
    print(f"Converted: {csv_file.name}")
```

