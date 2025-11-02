# Simple Download Instructions

## SharePoint Authentication Required

SharePoint requires authentication to download files programmatically.
For now, the **easiest approach is manual download**:

### Manual Download Steps:

1. **Open SharePoint folder in browser** (you're already logged in)
   ```
   https://ndusbpos.sharepoint.com/:f:/s/UNDBMEProjectSled-Head/EuiUFvmvn9lJiTW8dH4WJfsBZ5kjfMx5hMTquTz-vJTh3Q?e=i1Fx1f
   ```

2. **Download all files**:
   - Select all files (Ctrl/Cmd+A)
   - Click "Download"
   - OR download individual files

3. **Save to `data/00_collect/imu_raw/`**

4. **Convert XLSX to CSV** (run this script):
   ```python
   import pandas as pd
   from pathlib import Path
   
   xlsx_dir = Path('data/00_collect/imu_raw')
   
   for xlsx_file in xlsx_dir.glob('*.xlsx'):
       print(f"Converting: {xlsx_file.name}")
       df = pd.read_excel(xlsx_file)
       csv_file = xlsx_file.with_suffix('.csv')
       df.to_csv(csv_file, index=False)
       print(f"  ✓ Saved: {csv_file.name}")
   ```

5. **Delete XLSX files** (optional):
   ```bash
   rm data/00_collect/imu_raw/*.xlsx
   ```

---

### OR: Use OneDrive Sync

If you have OneDrive sync:
1. Sync the SharePoint folder to your OneDrive
2. Copy files from OneDrive → `data/00_collect/imu_raw/`
3. Convert to CSV using the script above

---

### Automated Download (Future)

To set up automated download, you'll need:
1. Azure AD app registration
2. OAuth tokens
3. Graph API access

For now, manual download is fastest!

