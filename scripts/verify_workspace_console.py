"""End-to-end check against an isolated local preview server, not production."""

import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    output = Path("/tmp/speedinfer-console-qa")
    output.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:8011/app")
        page.click("#tab-btn-register")
        page.fill("#register-name", "Sam")
        page.fill("#register-email", f"console-{time.time_ns()}@example.test")
        page.fill("#register-password", "ConsolePreview123!")
        page.check("#register-policy-consent")
        page.click("#register-submit-btn")
        page.wait_for_selector(".ws-hero")
        assert page.locator("#modal-secret-key-reveal.active").count() == 0
        page.locator("#toast-container .btn-icon").evaluate_all(
            "(buttons) => buttons.forEach(button => button.click())"
        )
        page.screenshot(path=str(output / "overview-desktop.png"), full_page=True)
        page.click('.nav-item[data-view="projects"]')
        page.fill('[data-simple="project"] input', "Arabic assistant")
        page.click('[data-simple="project"] button')
        page.wait_for_url("**/app/projects/*")
        page.click('.nav-item[data-view="buckets"]')
        page.fill('[data-simple="bucket"] input', "research-assets")
        page.click('[data-simple="bucket"] button')
        page.wait_for_selector("[data-upload]")
        page.set_input_files(
            'input[type="file"]',
            {
                "name": "support.jsonl",
                "mimeType": "application/json",
                "buffer": b'{"text":"How can I help you?"}\n',
            },
        )
        page.click('[data-upload] button[type="submit"]')
        page.wait_for_selector('[data-file-name="support.jsonl"]')
        page.select_option('select[name="purpose"]', "weights")
        page.set_input_files(
            'input[type="file"]',
            {
                "name": "sample.safetensors",
                "mimeType": "application/octet-stream",
                "buffer": b"preview only, not real model weights",
            },
        )
        page.click('[data-upload] button[type="submit"]')
        page.wait_for_selector('[data-file-name="sample.safetensors"]')
        page.screenshot(path=str(output / "bucket-desktop.png"), full_page=True)
        page.click('.nav-item[data-view="training"]')
        page.click('[data-go="training/new"]')
        page.wait_for_selector("[data-workflow-form]")
        page.fill('input[name="name"]', "Support QLoRA")
        page.select_option('select[name="project_id"]', index=1)
        page.select_option('select[name="dataset_id"]', index=1)
        page.select_option('select[name="weights_id"]', index=1)
        page.check('[data-workflow-form] input[type="checkbox"]')
        page.screenshot(path=str(output / "training-desktop.png"), full_page=True)
        page.click('[data-workflow-form] button[type="submit"]')
        page.wait_for_selector('[data-action="start"]')
        page.click('[data-action="start"]')
        page.wait_for_selector('[data-action="complete"]')
        page.click('[data-action="complete"]')
        page.wait_for_selector(".ws-badge.succeeded")
        page.reload()
        page.wait_for_selector(".ws-badge.succeeded")
        page.click('.nav-item[data-view="deployments"]')
        page.click('[data-go="deployments/new"]')
        page.wait_for_selector('[data-workflow-form="deployment"]')
        page.fill('input[name="name"]', "Support endpoint")
        page.select_option('select[name="project_id"]', index=1)
        page.select_option('select[name="artifact_id"]', index=1)
        page.check('[data-workflow-form] input[type="checkbox"]')
        page.click('[data-workflow-form] button[type="submit"]')
        page.wait_for_selector('[data-action="start"]')
        page.click('[data-action="start"]')
        page.wait_for_selector('[data-action="stop"]')
        page.click('[data-action="stop"]')
        page.wait_for_selector(".ws-badge.stopped")
        page.click('.nav-item[data-view="pricing"]')
        page.wait_for_selector(".ws-price")
        page.screenshot(path=str(output / "pricing-desktop.png"), full_page=True)
        page.click('.nav-item[data-view="keys"]')
        page.wait_for_selector("#keys-empty-state", state="visible")
        page.click("#open-create-key-modal-btn")
        page.fill("#new-key-name", "my-first-key")
        page.click("#create-key-submit-btn")
        page.wait_for_selector("#modal-secret-key-reveal.active")
        assert page.evaluate("sessionStorage.getItem('speedinfer_last_raw_key')") is None
        page.click("#modal-secret-key-reveal .modal-close-btn")
        assert page.locator("#modal-secret-key-val").inner_text() == ""
        page.click('.nav-item[data-view="dashboard"]')
        page.wait_for_selector(".ws-hero")
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_function(
            "document.querySelector('.app-sidebar').getBoundingClientRect().right <= 1"
        )
        page.locator("#toast-container .btn-icon").evaluate_all(
            "(buttons) => buttons.forEach(button => button.click())"
        )
        page.screenshot(path=str(output / "overview-mobile.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.click(".mobile-menu-btn")
        page.click('.nav-item[data-view="buckets"]')
        page.wait_for_selector('[data-simple="bucket"]')
        page.wait_for_function(
            "document.querySelector('.app-sidebar').getBoundingClientRect().right <= 1"
        )
        page.screenshot(path=str(output / "buckets-mobile.png"), full_page=True)
        page.goto("http://127.0.0.1:8011/legal/privacy")
        page.wait_for_selector(".ws-policy-body")
        assert "Account information" in page.locator(".ws-policy-body").inner_text()
        assert not errors, errors
        print(f"Workspace browser checks passed. Screenshots: {output}")
        browser.close()


if __name__ == "__main__":
    main()
