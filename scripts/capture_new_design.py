"""Capture screenshots of the new SpeedInfer design across landing and console."""

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ARTIFACT_DIR = Path(
    "/home/samimahmoud/.gemini/antigravity/brain/cd3fde06-4a52-4bc5-a168-b3e3dc533f90"
)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Start uvicorn server
cmd = [
    sys.executable,
    "-m",
    "uvicorn",
    "speedinfer.gateway.app:app",
    "--host",
    "127.0.0.1",
    "--port",
    "8000",
]
server_proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

ready = False
for _ in range(30):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1) as resp:
            if resp.status == 200:
                ready = True
                break
    except Exception:
        time.sleep(0.5)

if not ready:
    print("Server failed to start within timeout!")
    server_proc.terminate()
    stdout, stderr = server_proc.communicate()
    print("STDOUT:", stdout.decode())
    print("STDERR:", stderr.decode())
    sys.exit(1)

print("Server is healthy and ready!")

try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        # Desktop View - Landing Page Top
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto("http://127.0.0.1:8000", wait_until="networkidle")
        time.sleep(0.5)
        page.screenshot(path=str(ARTIFACT_DIR / "new_landing_hero.png"))

        # Scroll down to platform and technology sections
        page.evaluate("window.scrollTo(0, 1100)")
        time.sleep(0.5)
        page.screenshot(path=str(ARTIFACT_DIR / "new_landing_features.png"))

        # Scroll down to code and contact sections
        page.evaluate("window.scrollTo(0, 2400)")
        time.sleep(0.5)
        page.screenshot(path=str(ARTIFACT_DIR / "new_landing_developers.png"))

        # Mobile View - Landing Page
        page_mobile = browser.new_page(viewport={"width": 390, "height": 844})
        page_mobile.goto("http://127.0.0.1:8000", wait_until="networkidle")
        time.sleep(0.5)
        page_mobile.screenshot(path=str(ARTIFACT_DIR / "new_landing_mobile.png"))

        # Test Opening Developer Console via "Log in" or "Start building"
        page.click("button[data-auth='login']")
        time.sleep(0.5)
        page.screenshot(path=str(ARTIFACT_DIR / "new_auth_login.png"))

        # Login with test account
        page.fill("#login-email", "test@speedinfer.local")
        page.fill("#login-password", "Password123!")
        # Or register if test doesn't have password
        page.click("#tab-btn-register")
        time.sleep(0.3)
        unique_email = f"designer_{int(time.time())}@speedinfer.local"
        page.fill("#register-name", "Design Reviewer")
        page.fill("#register-email", unique_email)
        page.fill("#register-password", "Password123!*")
        page.click("#register-submit-btn")
        time.sleep(1.5)

        # Close secret key reveal modal if open
        if page.is_visible("#modal-secret-key-reveal.active"):
            page.screenshot(path=str(ARTIFACT_DIR / "new_modal_key_reveal.png"))
            page.click("#modal-secret-key-reveal .btn-secondary")
            time.sleep(0.5)

        # Dashboard View in New Design
        page.wait_for_selector("#view-dashboard", state="visible")
        time.sleep(0.5)
        page.screenshot(path=str(ARTIFACT_DIR / "new_dashboard.png"))

        # Playground View in New Design
        page.click(".nav-item[data-view='playground']")
        time.sleep(0.5)
        page.screenshot(path=str(ARTIFACT_DIR / "new_playground.png"))

        # Contact Dialog
        page.goto("http://127.0.0.1:8000#contact")
        time.sleep(0.5)
        if page.is_visible("button[data-contact='sales']"):
            page.click("button[data-contact='sales']")
            time.sleep(0.5)
            page.screenshot(path=str(ARTIFACT_DIR / "new_contact_modal.png"))

        browser.close()
    print("All new design screenshots captured successfully!")
finally:
    server_proc.terminate()
    server_proc.wait()
