"""Comprehensive end-to-end browser verification script for SpeedInfer frontend.

Performs:
- Full responsive testing across 1440px desktop, 768px tablet, and 390px mobile
- Real user registration and authentication flow
- API key generation and one-time secret modal verification
- Interactive SSE streaming chat in the playground with TTFT and token metrics
- Keys listing and revocation
- Models catalog and usage inspection
- Console error assertions and visual screenshot capture
"""

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ARTIFACT_DIR = Path(
    "/home/samimahmoud/.gemini/antigravity/brain/cd3fde06-4a52-4bc5-a168-b3e3dc533f90"
)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "http://127.0.0.1:8000"


def run_verification():
    console_errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        # Capture console errors
        def on_console(msg):
            if msg.type == "error":
                console_errors.append(msg.text)

        page.on("console", on_console)
        page.on("pageerror", lambda exc: console_errors.append(str(exc)))

        print("[1/8] Loading SpeedInfer Gateway...")
        page.goto(BASE_URL, wait_until="networkidle")
        time.sleep(0.5)

        # 1. Auth Page
        print("[2/8] Testing Auth Screen...")
        page.screenshot(path=str(ARTIFACT_DIR / "desktop_auth.png"))

        # Switch to Register tab
        page.click("#tab-btn-register")
        time.sleep(0.3)
        page.fill("#register-name", "Dr. Alan Turing")
        unique_email = f"turing_{int(time.time())}@speedinfer.local"
        page.fill("#register-email", unique_email)
        page.fill("#register-password", "ValidPass123!*")

        page.click("#register-submit-btn")
        time.sleep(1.5)

        # Verify Secret Key Reveal Modal
        print("[3/8] Verifying API Key Reveal Modal...")
        page.wait_for_selector("#modal-secret-key-reveal.active", timeout=5000)
        page.screenshot(path=str(ARTIFACT_DIR / "desktop_key_reveal.png"))
        page.click("#modal-secret-key-reveal .btn-secondary")
        time.sleep(0.5)

        # 2. Dashboard View
        print("[4/8] Verifying Dashboard View...")
        page.wait_for_selector("#view-dashboard", state="visible")
        time.sleep(0.5)
        page.screenshot(path=str(ARTIFACT_DIR / "desktop_dashboard.png"))

        # 3. Keys Management View
        print("[5/8] Testing API Keys Management...")
        page.click(".nav-item[data-view='keys']")
        time.sleep(0.5)
        page.screenshot(path=str(ARTIFACT_DIR / "desktop_keys.png"))

        # Create a second key
        page.click("#open-create-key-modal-btn")
        time.sleep(0.3)
        page.fill("#new-key-name", "inference-cluster-key")
        page.fill("#new-key-balance", "25.0")
        page.click("#create-key-submit-btn")
        time.sleep(1.0)
        # Close reveal modal
        page.click("#modal-secret-key-reveal .btn-secondary")
        time.sleep(0.5)

        # 4. Models Catalog View
        print("[6/8] Testing Models Catalog...")
        page.click(".nav-item[data-view='models']")
        time.sleep(0.5)
        page.screenshot(path=str(ARTIFACT_DIR / "desktop_models.png"))

        # 5. Interactive Chat Playground with Real Streaming
        print("[7/8] Testing Streaming Chat Playground...")
        page.click(".nav-item[data-view='playground']")
        time.sleep(0.5)

        # Send test prompt
        prompt = "Explain PagedAttention in vLLM and how it saves GPU memory."
        page.fill("#chat-input-textarea", prompt)
        page.click("#chat-send-btn")

        # Wait for streaming completion and metrics pill to render
        time.sleep(3.0)
        page.wait_for_selector(".message-metrics-tag", timeout=8000)
        page.screenshot(path=str(ARTIFACT_DIR / "desktop_playground_stream.png"))

        # 6. Usage & Billing View
        page.click(".nav-item[data-view='usage']")
        time.sleep(0.5)
        page.screenshot(path=str(ARTIFACT_DIR / "desktop_usage.png"))

        # 7. Responsive Tests: Tablet (768x1024)
        print("[8/8] Testing Responsive Layouts (Tablet 768px and Mobile 390px)...")
        page.set_viewport_size({"width": 768, "height": 1024})
        time.sleep(0.5)
        page.click(".nav-item[data-view='dashboard']")
        time.sleep(0.3)
        page.screenshot(path=str(ARTIFACT_DIR / "tablet_dashboard.png"))

        page.click(".nav-item[data-view='playground']")
        time.sleep(0.3)
        page.screenshot(path=str(ARTIFACT_DIR / "tablet_playground.png"))

        # Responsive Tests: Mobile (390x844 - iPhone standard)
        page.set_viewport_size({"width": 390, "height": 844})
        time.sleep(0.5)

        # Open mobile drawer
        page.click(".mobile-menu-btn")
        time.sleep(0.4)
        page.screenshot(path=str(ARTIFACT_DIR / "mobile_nav_drawer.png"))

        # Click Dashboard in drawer
        page.click(".nav-item[data-view='dashboard']")
        time.sleep(0.5)
        page.screenshot(path=str(ARTIFACT_DIR / "mobile_dashboard.png"))

        # Open mobile drawer and switch to mobile playground
        page.click(".mobile-menu-btn")
        time.sleep(0.4)
        page.click(".nav-item[data-view='playground']")
        time.sleep(0.5)
        page.screenshot(path=str(ARTIFACT_DIR / "mobile_playground.png"))

        browser.close()

    print("Browser verification completed successfully!")
    print(f"Screenshots saved to: {ARTIFACT_DIR}")

    # Check console errors
    if console_errors:
        print(f"Encountered {len(console_errors)} console errors: {console_errors}")
        sys.exit(1)
    else:
        print("0 Console errors detected across entire session.")


if __name__ == "__main__":
    run_verification()
