"""Browser smoke tests. Run against a live server: E2E_BASE_URL=http://127.0.0.1:8081 python -m unittest discover -s tests/e2e"""
import os
import unittest
from pathlib import Path

BASE = os.environ.get("E2E_BASE_URL")


@unittest.skipUnless(BASE, "set E2E_BASE_URL to run browser tests")
class BrowserSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch(args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def open(self, path, wait_until="networkidle", **ctx):
        context = self.browser.new_context(**ctx)
        self.addCleanup(context.close)
        page = context.new_page()
        self.errors = []
        page.on("pageerror", lambda e: self.errors.append(str(e)))
        page.on("console", lambda m: m.type == "error" and self.errors.append(m.text))
        page.goto(BASE + path, wait_until=wait_until)
        return page

    def kpi(self, page, element_id):
        return page.locator(f"#{element_id}").inner_text().strip()

    def test_dashboard_buttons_drive_the_line(self):
        page = self.open("/", wait_until="load", viewport={"width": 1440, "height": 900})  # dashboard polls, never idle
        page.wait_for_function("window.lineTwinReady === true", timeout=15000)
        page.get_by_role("button", name="Master Reset").click()
        page.get_by_role("button", name="Start Cycle").click()
        page.wait_for_function("document.getElementById('status_val').textContent === 'RUNNING'", timeout=5000)
        page.get_by_role("button", name="Simulate Defect").click()
        page.wait_for_function("document.querySelector('#log-tbody .text-fail') !== null", timeout=15000)
        self.assertIn("·", page.locator("#log-tbody .text-fail").first.inner_text())  # defect class shown
        self.assertGreaterEqual(int(self.kpi(page, "val_total")), 1)
        # Capture the real running HMI with an observed simulated rejection.
        target = os.environ.get("E2E_SCREENSHOT_DIR")
        if target:
            Path(target).mkdir(parents=True, exist_ok=True)
            page.screenshot(
                path=str(Path(target) / "visual-qc-desktop-1440.png"),
                full_page=True,
                animations="disabled",
            )
        page.get_by_role("button", name="Pause System").click()
        page.wait_for_function("document.getElementById('status_val').textContent === 'PAUSED'", timeout=5000)
        page.get_by_role("button", name="Emergency Stop").click()
        page.wait_for_function("document.getElementById('status_val').textContent === 'ESTOP'", timeout=5000)
        page.wait_for_selector("#twin-banner.is-estop", timeout=5000)
        with page.expect_download() as download:
            page.get_by_role("link", name="Export Report").click()
        self.assertTrue(download.value.suggested_filename.endswith(".xlsx"))
        self.assertEqual(self.errors, [])

    def test_estop_latches_until_reset_in_real_hmi(self):
        page = self.open("/", wait_until="load", viewport={"width": 1280, "height": 900})
        page.get_by_role("button", name="Master Reset").click()
        page.get_by_role("button", name="Start Cycle").click()
        page.wait_for_function("document.getElementById('status_val').textContent === 'RUNNING'", timeout=7000)
        page.get_by_role("button", name="Emergency Stop").click()
        page.wait_for_function("document.getElementById('status_val').textContent === 'ESTOP'", timeout=7000)
        page.get_by_role("button", name="Start Cycle").click()
        page.wait_for_function("document.getElementById('nexus-command-label').textContent.includes('RESET REQUIRED')", timeout=7000)
        self.assertEqual(page.locator("#status_val").inner_text(), "ESTOP")
        page.get_by_role("button", name="Master Reset").click()
        page.wait_for_function("document.getElementById('status_val').textContent === 'PAUSED'", timeout=7000)
        page.get_by_role("button", name="Start Cycle").click()
        page.wait_for_function("document.getElementById('status_val').textContent === 'RUNNING'", timeout=7000)
        self.assertEqual(self.errors, [])

    def test_nexus_mobile_hmi_panels_are_reachable(self):
        page = self.open("/", wait_until="load", viewport={"width": 375, "height": 812},
                         is_mobile=True, has_touch=True)
        page.locator("#nexus-command-status").wait_for()
        page.wait_for_function("!document.getElementById('val_total').hasAttribute('data-loading')", timeout=8000)
        self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), 376)
        for selector in ("#line-twin", "#profitChart", "#oeeChart", "#log-tbody"):
            item = page.locator(selector)
            item.scroll_into_view_if_needed()
            self.assertTrue(item.is_visible(), selector)
        # The two HUD chips must not cover each other at 375px.
        page.locator('.twin-tl').wait_for(timeout=15000)
        title_bounds = page.locator('.twin-tl').bounding_box()
        status_bounds = page.locator('.twin-tr').bounding_box()
        self.assertIsNotNone(title_bounds)
        self.assertIsNotNone(status_bounds)
        self.assertLessEqual(title_bounds['y'] + title_bounds['height'], status_bounds['y'] + 1)
        page.get_by_role('button', name='Start Cycle').click()
        page.wait_for_function("document.getElementById('status_val').textContent === 'RUNNING'", timeout=7000)
        page.wait_for_function("Number(document.getElementById('val_total').textContent) >= 1", timeout=15000)
        page.evaluate('window.scrollTo(0,0)')
        self.assertEqual(self.errors, [])

        # Optional real screenshots (not generated artwork). Disabled unless a
        # reviewer explicitly selects a local output directory.
        target = os.environ.get("E2E_SCREENSHOT_DIR")
        if target:
            Path(target).mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(Path(target) / "visual-qc-mobile-375.png"), full_page=True)

    def test_rag_loads_three_only_on_scroll_and_switches_questions(self):
        page = self.open("/rag", viewport={"width": 1280, "height": 420})  # map well below the fold
        loaded = r"performance.getEntriesByType('resource').some(e => /three\.module/.test(e.name))"
        self.assertGreater(page.evaluate("document.getElementById('map-frame').getBoundingClientRect().top - innerHeight"), 200)
        self.assertFalse(page.evaluate(loaded))
        page.locator("#map-section").scroll_into_view_if_needed()
        page.wait_for_selector("#map-frame canvas", timeout=15000)
        self.assertTrue(page.evaluate(loaded))
        page.locator(".q-btn").nth(3).click()
        page.wait_for_function("document.querySelector('.subhead .badge').textContent.toLowerCase().includes('declined')")
        self.assertEqual(self.errors, [])

    def test_rag_mobile_uses_2d_without_horizontal_scroll(self):
        page = self.open("/rag", viewport={"width": 375, "height": 812}, is_mobile=True, has_touch=True)
        page.locator("#map-section").scroll_into_view_if_needed()
        page.wait_for_selector("#map-frame.is-2d canvas", timeout=10000)
        self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), 375)

    def test_case_study_is_interactive(self):
        page = self.open("/case-study", viewport={"width": 1280, "height": 800})
        page.wait_for_function("document.querySelectorAll('.cm .cell').length === 36")
        page.locator(".queue").scroll_into_view_if_needed()
        page.get_by_role("button", name="Review 10%").click()
        page.wait_for_function("document.getElementById('q-caption').textContent.startsWith('10 of 22')")
        self.assertEqual(page.locator("#waffle i.hit").count(), 10)
        self.assertEqual(self.errors, [])


if __name__ == "__main__":
    unittest.main()
