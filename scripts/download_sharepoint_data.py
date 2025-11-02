"""Script to download IMU data files from SharePoint.

This script downloads Excel files from SharePoint and converts them to CSV format
for use in the Sled-Head IMU pipeline.

Requirements:
    pip install Office365-REST-Python-Client openpyxl pandas requests

Usage:
    python scripts/download_sharepoint_data.py
    
Environment variables:
    SHAREPOINT_URL - Your SharePoint site URL
    SHAREPOINT_USERNAME - Your email/username
    SHAREPOINT_PASSWORD - Your password (or use credential manager)
"""

import os
import sys
from pathlib import Path
from typing import List
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def download_with_requests_sharepoint(
    sharepoint_url: str,
    folder_path: str,
    username: str,
    password: str,
    output_dir: Path,
    convert_xlsx: bool = True
) -> List[Path]:
    """Download files from SharePoint using SharePoint API.
    
    Args:
        sharepoint_url: SharePoint site URL
        folder_path: Folder path in SharePoint
        username: SharePoint username
        password: SharePoint password
        output_dir: Local directory to save files
        convert_xlsx: If True, convert .xlsx to .csv
        
    Returns:
        List of downloaded file paths
    """
    try:
        from office365.sharepoint.client_context import ClientContext
        from office365.runtime.auth.authentication_context import AuthenticationContext
    except ImportError:
        print("ERROR: office365 package not installed.")
        print("Install with: pip install Office365-REST-Python-Client")
        return []
    
    try:
        # Authenticate
        ctx_auth = AuthenticationContext(sharepoint_url)
        if ctx_auth.acquire_token_for_user(username, password):
            ctx = ClientContext(sharepoint_url, ctx_auth)
            
            # Get folder
            folder = ctx.web.get_folder_by_server_relative_url(folder_path)
            ctx.load(folder)
            ctx.execute_query()
            
            # Get files
            files = folder.files
            ctx.load(files)
            ctx.execute_query()
            
            downloaded_files = []
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for file in files:
                print(f"Downloading: {file.properties['Name']}")
                
                # Download file
                file_content = file.read()
                local_path = output_dir / file.properties['Name']
                
                with open(local_path, 'wb') as local_file:
                    local_file.write(file_content)
                
                downloaded_files.append(local_path)
                
                # Convert XLSX to CSV
                if convert_xlsx and local_path.suffix.lower() == '.xlsx':
                    print(f"  Converting to CSV...")
                    df = pd.read_excel(local_path)
                    csv_path = local_path.with_suffix('.csv')
                    df.to_csv(csv_path, index=False)
                    print(f"  ✓ Saved: {csv_path.name}")
                    downloaded_files.append(csv_path)
                    # Remove XLSX file
                    local_path.unlink()
            
            print(f"\n✓ Downloaded {len(downloaded_files)} files to {output_dir}")
            return downloaded_files
            
        else:
            print("ERROR: Failed to authenticate with SharePoint")
            return []
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return []


def download_with_shareplum(
    sharepoint_url: str,
    folder_name: str,
    username: str,
    password: str,
    output_dir: Path,
    convert_xlsx: bool = True
) -> List[Path]:
    """Download files using SharePlum library.
    
    Args:
        sharepoint_url: SharePoint site URL
        folder_name: Folder name in SharePoint
        username: SharePoint username
        password: SharePoint password
        output_dir: Local directory to save files
        convert_xlsx: If True, convert .xlsx to .csv
        
    Returns:
        List of downloaded file paths
    """
    try:
        from shareplum import Site
        from shareplum.office365 import Office365
    except ImportError:
        print("ERROR: shareplum package not installed.")
        print("Install with: pip install shareplum requests-ntlm")
        return []
    
    try:
        # Login
        authcookie = Office365(sharepoint_url, username=username, password=password).GetCookies()
        site = Site(sharepoint_url, authcookie=authcookie)
        
        # Get folder
        folder = site.Folder(folder_name)
        files = folder.files
        
        downloaded_files = []
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for file_info in files:
            filename = file_info['Name']
            print(f"Downloading: {filename}")
            
            file_content = folder.get_file(filename)
            local_path = output_dir / filename
            
            with open(local_path, 'wb') as f:
                f.write(file_content)
            
            downloaded_files.append(local_path)
            
            # Convert XLSX to CSV
            if convert_xlsx and local_path.suffix.lower() == '.xlsx':
                print(f"  Converting to CSV...")
                df = pd.read_excel(local_path)
                csv_path = local_path.with_suffix('.csv')
                df.to_csv(csv_path, index=False)
                print(f"  ✓ Saved: {csv_path.name}")
                downloaded_files.append(csv_path)
                # Remove XLSX file
                local_path.unlink()
        
        print(f"\n✓ Downloaded {len(downloaded_files)} files to {output_dir}")
        return downloaded_files
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return []


def convert_sharepoint_url(sharepoint_url: str) -> str:
    """Convert SharePoint folder/file URL to direct download URL.
    
    Converts URLs like:
        https://...sharepoint.com/:f:/s/.../EuiUF...?e=abc123
    to:
        https://...sharepoint.com/sites/.../_layouts/15/download.aspx?UniqueId=...
    
    Args:
        sharepoint_url: Original SharePoint URL
        
    Returns:
        Direct download URL
    """
    # Extract key parts from SharePoint URL
    # Format: https://domain.sharepoint.com/:f:/s/sitename/key?e=token
    
    try:
        if '/:f:/' in sharepoint_url:
            # Folder URL - need to extract individual file IDs
            # This requires more complex parsing or API access
            print("⚠️  Folder URL detected. You may need to provide direct file URLs.")
            return sharepoint_url
        elif '/:x:/' in sharepoint_url or '/:o:/' in sharepoint_url:
            # File URL format
            parts = sharepoint_url.split('?')[0].split('/')
            if 'e=' in sharepoint_url:
                # Try to construct download URL
                base_url = sharepoint_url.split('/:')[0]
                file_key = parts[-1]
                return f"{base_url}/download.aspx?UniqueId={file_key}"
        
        return sharepoint_url
        
    except Exception as e:
        print(f"Warning: Could not convert URL: {str(e)}")
        return sharepoint_url


def download_from_csv_urls(
    urls_file: str,
    output_dir: Path
) -> List[Path]:
    """Download files from a CSV containing SharePoint URLs.
    
    Alternative approach: Export the SharePoint file list to CSV,
    then use this function to download from URLs.
    
    Args:
        urls_file: Path to CSV file with URLs
        output_dir: Local directory to save files
        
    Returns:
        List of downloaded file paths
    """
    import requests
    
    urls_df = pd.read_csv(urls_file)
    
    if 'url' not in urls_df.columns or 'filename' not in urls_df.columns:
        print("ERROR: CSV must have 'url' and 'filename' columns")
        return []
    
    downloaded_files = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for _, row in urls_df.iterrows():
        url = row['url']
        filename = row['filename']
        
        print(f"Downloading: {filename}")
        
        try:
            # Try converting SharePoint URL to direct download
            direct_url = convert_sharepoint_url(url)
            
            response = requests.get(direct_url, stream=True, allow_redirects=True)
            response.raise_for_status()
            
            local_path = output_dir / filename
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            downloaded_files.append(local_path)
            print(f"  ✓ Saved: {filename}")
            
            # Convert XLSX to CSV if needed
            if local_path.suffix.lower() == '.xlsx':
                print(f"  Converting to CSV...")
                df = pd.read_excel(local_path)
                csv_path = local_path.with_suffix('.csv')
                df.to_csv(csv_path, index=False)
                downloaded_files.append(csv_path)
                local_path.unlink()
                
        except Exception as e:
            print(f"  ERROR: {str(e)}")
    
    print(f"\n✓ Downloaded {len(downloaded_files)} files")
    return downloaded_files


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download IMU data from SharePoint')
    parser.add_argument(
        '--url',
        help='SharePoint folder URL (if using SharePoint API)',
        default=os.getenv('SHAREPOINT_URL', '')
    )
    parser.add_argument(
        '--username',
        help='SharePoint username',
        default=os.getenv('SHAREPOINT_USERNAME', '')
    )
    parser.add_argument(
        '--password',
        help='SharePoint password',
        default=os.getenv('SHAREPOINT_PASSWORD', '')
    )
    parser.add_argument(
        '--csv-urls',
        help='CSV file with SharePoint URLs and filenames',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory for downloaded files',
        default='data/00_collect/imu_raw',
        type=str
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    print("="*80)
    print("SHAREPOINT IMU DATA DOWNLOADER")
    print("="*80)
    
    # If CSV with URLs provided, use that method
    if args.csv_urls:
        print("\n📁 Using CSV file with URLs...")
        download_from_csv_urls(args.csv_urls, output_dir)
        return
    
    # Otherwise, try SharePoint API
    if not args.url or not args.username:
        print("\n⚠️  SharePoint credentials not provided.")
        print("\nOptions:")
        print("  1. Provide credentials via --url and --username flags")
        print("  2. Create a CSV file with URLs and use --csv-urls")
        print("\nFor CSV method:")
        print("  - Export SharePoint file list to CSV with 'url' and 'filename' columns")
        print("  - Run: python scripts/download_sharepoint_data.py --csv-urls file.csv")
        return
    
    print(f"\n📁 SharePoint URL: {args.url}")
    print(f"👤 Username: {args.username}")
    
    # Try SharePlum method first
    print("\nAttempting download with SharePlum...")
    files = download_with_shareplum(
        args.url, 
        folder_name="",
        username=args.username,
        password=args.password or input("Password: "),
        output_dir=output_dir
    )
    
    if not files:
        print("\nSharePlum failed. Trying Office365-REST-Python-Client...")
        files = download_with_requests_sharepoint(
            args.url,
            folder_path="",
            username=args.username,
            password=args.password or input("Password: "),
            output_dir=output_dir
        )
    
    if files:
        print("\n✅ Download complete!")
    else:
        print("\n❌ Download failed. See errors above.")


if __name__ == "__main__":
    main()

