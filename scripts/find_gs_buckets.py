# This script is used to find the unique checkpoint names in a GCS bucket.
import sys
import json
import argparse
from urllib.parse import urlencode
import urllib.request

def extract_roots(items):
    roots = set()
    for obj in items:
        name = obj.get("name", "")
        if not name.startswith("checkpoints/"):
            continue
        parts = name.split("/")
        if len(parts) >= 2:
            roots.add(parts[1])
    return roots

def from_stdin():
    data = sys.stdin.read()
    if not data.strip():
        return set()
    payload = json.loads(data)
    items = payload.get("items", [])
    return extract_roots(items)

def from_url(url):
    roots = set()
    page_token = None
    while True:
        q = {"prefix": "checkpoints"}
        if page_token:
            q["pageToken"] = page_token
        full_url = url + ("&" if "?" in url else "?") + urlencode(q)
        with urllib.request.urlopen(full_url) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        items = payload.get("items", [])
        roots |= extract_roots(items)
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return roots

def main():
    ap = argparse.ArgumentParser(description="List unique checkpoint names from GCS listing JSON.")
    ap.add_argument("--url", default=None,
                    help="JSON API URL for the bucket, e.g. "
                         "'https://storage.googleapis.com/storage/v1/b/openpi-assets/o'")
    args = ap.parse_args()

    if args.url:
        roots = from_url(args.url)
    else:
        roots = from_stdin()

    for name in sorted(roots):
        print(name)

if __name__ == "__main__":
    main()
