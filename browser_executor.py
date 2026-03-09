"""
Playwright browser management for WebDreamer agent.
Handles navigation, screenshots, accessibility tree extraction, and action execution.
"""

import re
import base64
from io import BytesIO
from PIL import Image
from playwright.sync_api import sync_playwright, Page, Browser, Playwright

INTERACTIVE_ROLES = {
    "button", "link", "textbox", "checkbox", "radio", "combobox",
    "menuitem", "menuitemcheckbox", "menuitemradio",
    "searchbox", "slider", "spinbutton", "switch", "tab", "treeitem",
}
# "option" excluded — too noisy (e.g. dropdown options flood the tree)

MAX_ELEMENTS = 100


class BrowserManager:
    def __init__(self, headless=False):
        self._pw: Playwright = sync_playwright().start()
        self._browser: Browser = self._pw.chromium.launch(headless=headless)
        self._context = self._browser.new_context(viewport={"width": 1280, "height": 720})
        self._page: Page = self._context.new_page()

    def goto(self, url: str):
        self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
        self._page.wait_for_timeout(2000)

    def current_url(self) -> str:
        return self._page.url

    def screenshot_bytes(self) -> bytes:
        return self._page.screenshot(type="png")

    def screenshot_b64(self) -> str:
        raw = self.screenshot_bytes()
        b64 = base64.b64encode(raw).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def screenshot_pil(self) -> Image.Image:
        raw = self.screenshot_bytes()
        return Image.open(BytesIO(raw)).convert("RGB")

    def get_accessibility_tree(self) -> tuple[str, dict]:
        """Extract and flatten the accessibility tree using aria_snapshot.

        Returns:
            (formatted_string, id_map) where id_map maps int IDs to (role, name) tuples.
        """
        try:
            snapshot = self._page.locator("body").aria_snapshot()
        except Exception:
            return "", {}

        if not snapshot:
            return "", {}

        lines = []
        id_map = {}
        counter = 0

        for raw_line in snapshot.split("\n"):
            if counter >= MAX_ELEMENTS:
                break

            stripped = raw_line.strip()
            if not stripped or stripped.startswith("/"):
                continue

            # Strip leading "- " from YAML-like format
            cleaned = re.sub(r"^-\s*", "", stripped)

            # Parse "role "name"" or "role "name": content" patterns
            m = re.match(r'^(\w+)\s+"([^"]*)"', cleaned)
            if not m:
                continue

            role = m.group(1).lower()
            name = m.group(2)

            if role in INTERACTIVE_ROLES:
                id_map[counter] = (role, name)
                lines.append(f"[{counter}] {role} \"{name}\"")
                counter += 1

        return "\n".join(lines), id_map

    def execute(self, action_type: str, element_id: int | None,
                value: str | None, id_map: dict) -> str:
        """Execute an action in the browser. Returns status string."""
        action_type = action_type.lower().strip()

        if action_type == "stop":
            return "STOP"

        if action_type in ("scroll down", "scroll_down"):
            self._page.mouse.wheel(0, 500)
            self._wait()
            return "scrolled down"

        if action_type in ("scroll up", "scroll_up"):
            self._page.mouse.wheel(0, -500)
            self._wait()
            return "scrolled up"

        if action_type == "goto" and value:
            self._page.goto(value, wait_until="domcontentloaded", timeout=15000)
            self._wait()
            return f"navigated to {value}"

        if action_type == "go_back":
            self._page.go_back(wait_until="domcontentloaded", timeout=15000)
            self._wait()
            return "went back"

        if action_type == "go_forward":
            self._page.go_forward(wait_until="domcontentloaded", timeout=15000)
            self._wait()
            return "went forward"

        if action_type == "press" and value:
            self._page.keyboard.press(value)
            self._wait()
            return f"pressed {value}"

        # Actions that need an element
        if element_id is None or element_id not in id_map:
            return f"error: no valid element for {action_type}"

        locator = self._resolve_element(element_id, id_map)
        if locator is None:
            return f"error: could not find element [{element_id}]"

        try:
            if action_type == "click":
                locator.click(timeout=5000)
            elif action_type == "type":
                locator.click(timeout=5000)
                locator.fill(value or "", timeout=5000)
                # Auto-submit search boxes
                role, _name = id_map.get(element_id, ("", ""))
                if role.lower() in ("searchbox", "textbox"):
                    self._page.keyboard.press("Enter")
            elif action_type == "hover":
                locator.hover(timeout=5000)
            else:
                return f"error: unknown action type '{action_type}'"
        except Exception as e:
            return f"error: {e}"

        self._wait()
        return f"executed {action_type} on [{element_id}]"

    def _resolve_element(self, element_id: int, id_map: dict):
        """Resolve an element ID to a Playwright locator."""
        role, name = id_map[element_id]
        try:
            locator = self._page.get_by_role(role, name=name).first
            if locator.count() > 0:
                return locator
        except Exception:
            pass

        # Fallback: try by text
        if name:
            try:
                locator = self._page.get_by_text(name, exact=False).first
                if locator.count() > 0:
                    return locator
            except Exception:
                pass

        return None

    def _wait(self):
        """Wait for page to settle after an action."""
        try:
            self._page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass
        self._page.wait_for_timeout(1000)

    def close(self):
        self._browser.close()
        self._pw.stop()


if __name__ == "__main__":
    # Quick self-test
    bm = BrowserManager(headless=True)
    bm.goto("https://example.com")
    print(f"URL: {bm.current_url()}")

    tree_str, id_map = bm.get_accessibility_tree()
    print(f"\nAccessibility tree:\n{tree_str}")
    print(f"\nID map: {id_map}")

    pil = bm.screenshot_pil()
    print(f"\nScreenshot size: {pil.size}")

    bm.close()
    print("\nSelf-test passed.")
