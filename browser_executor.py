"""
Playwright browser management for WebDreamer agent.
Handles navigation, screenshots, accessibility tree extraction, and action execution.
"""

import re
import random
import base64
from io import BytesIO
from PIL import Image
from playwright.sync_api import sync_playwright, Page, Browser, Playwright
from playwright_stealth import Stealth

USER_AGENTS = {
    "firefox": [
        "Mozilla/5.0 (X11; Linux x86_64; rv:137.0) Gecko/20100101 Firefox/137.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:136.0) Gecko/20100101 Firefox/136.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:135.0) Gecko/20100101 Firefox/135.0",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:134.0) Gecko/20100101 Firefox/134.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
    ],
    "chromium": [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    ],
    "webkit": [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.15",
    ],
}

INTERACTIVE_ROLES = {
    "button", "link", "textbox", "checkbox", "radio", "combobox",
    "menuitem", "menuitemcheckbox", "menuitemradio",
    "searchbox", "slider", "spinbutton", "switch", "tab", "treeitem",
}
# "option" excluded — too noisy (e.g. dropdown options flood the tree)

MAX_ELEMENTS = 100


class BrowserManager:
    def __init__(self, headless=False, browser_type="chromium"):
        self._pw: Playwright = sync_playwright().start()
        engines = {
            "chromium": self._pw.chromium,
            "firefox": self._pw.firefox,
            "webkit": self._pw.webkit,
        }
        engine = engines.get(browser_type, self._pw.chromium)
        self._browser: Browser = engine.launch(headless=headless)
        self._context = self._browser.new_context(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=1,
            user_agent=random.choice(USER_AGENTS.get(browser_type, USER_AGENTS["firefox"])),
        )
        stealth = Stealth()
        self._page: Page = self._context.new_page()
        stealth.apply_stealth_sync(self._page)

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
        # Resolve placeholder IDs (e.g. "?search_box") by fuzzy matching
        if isinstance(element_id, str) and element_id.startswith("?"):
            element_id = self._resolve_placeholder(element_id, id_map)
            if element_id is None:
                return f"error: could not resolve placeholder to any element"

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
                # Auto-submit search/input boxes
                role, _name = id_map.get(element_id, ("", ""))
                if role.lower() in ("searchbox", "textbox", "combobox"):
                    self._page.keyboard.press("Enter")
            elif action_type == "hover":
                locator.hover(timeout=5000)
            else:
                return f"error: unknown action type '{action_type}'"
        except Exception as e:
            return f"error: {e}"

        self._wait()
        return f"executed {action_type} on [{element_id}]"

    def _resolve_placeholder(self, placeholder: str, id_map: dict) -> int | None:
        """Resolve a placeholder like '?search_box' to a real element ID.

        Asks the LLM to pick the best matching element from the current page.
        """
        from llm_call import call_text_llm

        desc = placeholder.lstrip("?").replace("_", " ")

        elements_str = "\n".join(
            f"[{eid}] {role} \"{name}\"" for eid, (role, name) in id_map.items()
        )

        system = "You match placeholder element descriptions to real page elements. Reply with ONLY the numeric ID number, nothing else."
        prompt = f"""The plan references an element called "{desc}".

Here are the actual interactive elements on the current page:
{elements_str}

Which element ID best matches "{desc}"? Reply with ONLY the number."""

        try:
            result = call_text_llm(system, prompt, max_tokens=16).strip()
            # Extract first number from response
            import re
            m = re.search(r'\d+', result)
            if m:
                eid = int(m.group())
                if eid in id_map:
                    role, name = id_map[eid]
                    print(f"  Resolved [{placeholder}] -> [{eid}] {role} \"{name}\"")
                    return eid
        except Exception as e:
            print(f"  Placeholder resolution error: {e}")

        print(f"  Could not resolve placeholder [{placeholder}]")
        return None

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
