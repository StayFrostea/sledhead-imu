"""Test SharePoint connection to verify access."""

import requests
from pathlib import Path

# Test the SharePoint folder URL
sharepoint_url = "https://ndusbpos.sharepoint.com/:f:/s/UNDBMEProjectSled-Head/EuiUFvmvn9lJiTW8dH4WJfsBZ5kjfMx5hMTquTz-vJTh3Q?e=i1Fx1f"

print("="*80)
print("TESTING SHAREPOINT CONNECTION")
print("="*80)
print(f"\nURL: {sharepoint_url}")

# Try to access the folder
print("\nAttempting to access SharePoint folder...")

try:
    response = requests.get(sharepoint_url, allow_redirects=True)
    print(f"Status Code: {response.status_code}")
    print(f"Final URL: {response.url}")
    
    if response.status_code == 200:
        print("✅ Successfully accessed SharePoint folder")
        
        # Check if it's HTML (web page) or something else
        content_type = response.headers.get('Content-Type', '')
        print(f"Content Type: {content_type}")
        
        if 'html' in content_type.lower():
            print("\n⚠️  This is a web folder view, not a direct download.")
            print("You'll need to get individual file URLs from the folder.")
    else:
        print(f"⚠️  Got status {response.status_code}")
        
except Exception as e:
    print(f"❌ Error: {str(e)}")

print("\n" + "="*80)

