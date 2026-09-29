#!/usr/bin/env python3
"""Submit all SpeedInfer URLs to search engines via IndexNow and sitemap pings."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import httpx

HOST = "speedinfer.com"
KEY = "c3a8b2f91e4d07528e1c6b5a"
KEY_LOCATION = f"https://{HOST}/{KEY}.txt"
SITEMAP_URL = f"https://{HOST}/sitemap.xml"

def get_urls_from_sitemap() -> list[str]:
    sitemap_path = Path(__file__).resolve().parent.parent / "speedinfer" / "frontend" / "sitemap.xml"
    if not sitemap_path.exists():
        print(f"Error: {sitemap_path} does not exist.")
        return []
    tree = ET.parse(sitemap_path)
    root = tree.getroot()
    namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [loc.text.strip() for loc in root.findall(".//ns:loc", namespace) if loc.text]
    return urls

def submit_indexnow(urls: list[str]) -> bool:
    print(f"Submitting {len(urls)} URLs to IndexNow (Bing, Yandex, Seznam)...")
    payload = {
        "host": HOST,
        "key": KEY,
        "keyLocation": KEY_LOCATION,
        "urlList": urls,
    }
    endpoints = [
        "https://api.indexnow.org/indexnow",
        "https://www.bing.com/indexnow",
    ]
    success = True
    for ep in endpoints:
        try:
            with httpx.Client(timeout=10.0) as client:
                res = client.post(ep, json=payload, headers={"Content-Type": "application/json; charset=utf-8"})
                print(f"  POST {ep} -> HTTP {res.status_code} ({res.reason_phrase})")
                if res.status_code not in (200, 202):
                    success = False
        except Exception as exc:
            print(f"  Error submitting to {ep}: {exc}")
            success = False
    return success

def ping_sitemaps() -> None:
    print("\nPinging search engine sitemap submission endpoints...")
    pings = [
        ("Google", f"https://www.google.com/ping?sitemap={SITEMAP_URL}"),
        ("Bing", f"https://www.bing.com/ping?sitemap={SITEMAP_URL}"),
    ]
    with httpx.Client(timeout=10.0, follow_redirects=True) as client:
        for name, url in pings:
            try:
                res = client.get(url)
                print(f"  Ping {name}: HTTP {res.status_code}")
            except Exception as exc:
                print(f"  Ping {name} failed: {exc}")

def main():
    urls = get_urls_from_sitemap()
    print(f"Discovered {len(urls)} URLs in sitemap:")
    for u in urls:
        print(f"  - {u}")
    print()
    submit_indexnow(urls)
    ping_sitemaps()
    print("\nSubmission completed successfully!")

if __name__ == "__main__":
    main()
