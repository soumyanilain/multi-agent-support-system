"""
Playwright automation for the mock support ticket application.

The Requester Agent calls submit_ticket() only after the Specialist
Agent has returned a grounded result, so every value entered into the
form comes from the RAG pipeline rather than from hard-coded text.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_APP_PATH = BASE_DIR / "mock_support_app" / "index.html"

# Slow the browser down when it is visible, so the demo is watchable.
DEMO_SLOW_MO_MS = int(os.getenv("DEMO_SLOW_MO_MS", "500"))

# The form's <select> uses snake_case values, but the Specialist Agent
# returns the human-readable category label. This maps one to the other.
CATEGORY_TO_FORM_VALUE = {
    "Account Access": "account_access",
    "Hardware": "hardware",
    "Software": "software",
    "Network": "network",
}


class BrowserAutomationError(RuntimeError):
    """Raised when the browser workflow cannot be completed."""


def get_app_url():
    """
    Resolve the mock application to a file:// URL.

    MOCK_APP_PATH in .env wins; otherwise the copy inside the repo is
    used, which keeps the project runnable on any teammate's machine.
    """

    configured = os.getenv("MOCK_APP_PATH", "").strip()

    app_path = Path(configured) if configured else DEFAULT_APP_PATH

    app_path = app_path.expanduser().resolve()

    if not app_path.exists():
        raise BrowserAutomationError(
            f"Mock support application not found at: {app_path}\n"
            "Check MOCK_APP_PATH in your .env file."
        )

    return app_path.as_uri()


def to_form_value(category):
    """
    Translate a Specialist Agent category into a dropdown value.

    The knowledge base also declares Email and Security, which the
    form cannot accept. The Specialist Agent rejects those before we
    get here, so reaching this branch means the two category lists
    have drifted apart.
    """

    form_value = CATEGORY_TO_FORM_VALUE.get(category)

    if form_value is None:
        raise BrowserAutomationError(
            f"Category '{category}' has no matching option in the "
            "support form dropdown."
        )

    return form_value


def submit_ticket(question, result, headless=True):
    """
    Fill in and submit the support ticket form using the Specialist
    Agent's response, then verify the confirmation.

    Args:
        question: The user's original support request.
        result:   The Specialist Agent result. Requires "category"
                  and "resolution".
        headless: False shows the browser (used for the demo video).

    Returns:
        The ticket ID string on success, or None if the confirmation
        could not be verified.

    Raises:
        BrowserAutomationError: the app is missing, the category has
        no matching option, or the form reported a validation error.
    """

    category = result.get("category", "")
    resolution = result.get("resolution", "")

    if not resolution:
        raise BrowserAutomationError(
            "Specialist Agent result contained no resolution text."
        )

    form_value = to_form_value(category)

    app_url = get_app_url()

    print(f"Opening support application: {app_url}")

    with sync_playwright() as playwright:

        browser = playwright.chromium.launch(
            headless=headless,
            slow_mo=0 if headless else DEMO_SLOW_MO_MS,
        )

        page = browser.new_page()

        try:
            page.goto(app_url)

            # --- Fill the form using the Specialist Agent's answer ---

            print("Filling issue description...")
            page.fill("#issue", question)

            print(f"Selecting category: {category} ({form_value})")
            page.select_option("#category", form_value)

            print("Filling resolution notes...")
            page.fill("#resolution", resolution)

            print("Submitting ticket...")
            page.click("#submit-ticket")

            # --- Verify the outcome ---

            # The app reveals #confirmation by removing .hidden, and
            # shows #error instead when a field is empty.
            try:
                page.wait_for_selector(
                    "#confirmation:not(.hidden)",
                    timeout=5000,
                )

            except PlaywrightTimeoutError:

                error_visible = page.is_visible("#error")

                if error_visible:
                    message = page.inner_text("#error")
                    raise BrowserAutomationError(
                        f"The form rejected the submission: {message}"
                    )

                print("Confirmation did not appear.")
                return None

            ticket_id = page.inner_text("#ticket-id").strip()

            shown_category = page.inner_text("#ticket-category").strip()
            shown_resolution = page.inner_text("#ticket-resolution").strip()

            # The app generates a 5-digit ID, so this confirms we are
            # reading a real confirmation rather than a stale element.
            if not re.fullmatch(r"\d{5}", ticket_id):
                print(f"Unexpected ticket ID format: {ticket_id!r}")
                return None

            if shown_category != category:
                print(
                    "Confirmation category did not match: "
                    f"expected {category!r}, found {shown_category!r}"
                )
                return None

            if shown_resolution != resolution:
                print("Confirmation resolution did not match what was entered.")
                return None

            print(f"Verified ticket ID: {ticket_id}")

            if not headless:
                # Let the confirmation stay on screen for the video.
                page.wait_for_timeout(3000)

            return ticket_id

        finally:
            browser.close()


# ============================================================
# STANDALONE TEST
#
# Runs the browser workflow against a fixed result, without
# needing the Specialist Agent. Useful for developing and
# debugging the automation on its own.
#
#   python -m requester_agent.browser_agent
# ============================================================

if __name__ == "__main__":

    sample_question = (
        "I forgot my password and cannot log into my account."
    )

    sample_result = {
        "category": "Account Access",
        "resolution": (
            "Verify the user's identity using the approved identity "
            "verification procedure. Then reset the user's password "
            "and confirm that they can log in."
        ),
        "sources": ["password_reset.md", "account_access.md"],
        "confidence": 0.702,
    }

    ticket = submit_ticket(
        question=sample_question,
        result=sample_result,
        headless=False,
    )

    print(f"\nResult: {ticket}")