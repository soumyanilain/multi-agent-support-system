import argparse
import json
import os

from dotenv import load_dotenv

from requester_agent.a2a_client import A2AClient, SpecialistTaskFailed


load_dotenv()


SPECIALIST_URL = os.getenv(
    "SPECIALIST_URL",
    "http://127.0.0.1:8000"
)

POLL_INTERVAL = int(
    os.getenv("POLL_INTERVAL_SECONDS", "2")
)

TASK_TIMEOUT = int(
    os.getenv("TASK_TIMEOUT_SECONDS", "30")
)


def get_specialist_result(question):
    """
    A2A half of the workflow.

    1. Submit the question to the Specialist Agent
    2. Receive an immediate acknowledgment with a task ID
    3. Poll until completed / failed / timeout
    4. Return the Specialist Agent's result
    """

    client = A2AClient(base_url=SPECIALIST_URL)

    print("\n--- Requester Agent ---")
    print(f"User request: {question}")

    acknowledgment = client.submit_task(question)

    task_id = acknowledgment.get("task_id")
    status = acknowledgment.get("status")

    if not task_id:
        raise RuntimeError(
            "Specialist Agent did not return a task ID."
        )

    print("\nTask submitted successfully.")
    print(f"Task ID: {task_id}")
    print(f"Initial status: {status}")

    print("\nWaiting for Specialist Agent...")

    result = client.wait_for_result(
        task_id=task_id,
        timeout=TASK_TIMEOUT,
        poll_interval=POLL_INTERVAL
    )

    print("\nTask completed.")
    print("Specialist result:")
    print(json.dumps(result, indent=4))

    return result


def run_request(question, headless=True):
    """
    Full end-to-end workflow:
    A2A  ->  RAG result  ->  Playwright  ->  verification.

    The browser is only opened once a grounded result exists.
    Any failure or timeout aborts before automation begins.
    """

    result = get_specialist_result(question)

    from requester_agent.browser_agent import submit_ticket

    print("\n--- Playwright ---")

    ticket_id = submit_ticket(
        question=question,
        result=result,
        headless=headless
    )

    if ticket_id:
        print(f"\nSupport ticket submitted successfully. ID: {ticket_id}")
        return {"result": result, "ticket_id": ticket_id}

    print("\nTicket submission could not be verified.")
    return {"result": result, "ticket_id": None}


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Support System - Requester Agent"
    )

    parser.add_argument(
        "--question",
        help="Customer issue. Omit to be prompted."
    )

    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Run Playwright with a visible browser (for demos)."
    )

    args = parser.parse_args()

    print("==============================")
    print(" Multi-Agent Support System")
    print("==============================")

    question = args.question

    if not question:
        question = input("\nEnter the customer's issue: ").strip()

    if not question:
        print("Error: A question is required.")
        return

    try:
        run_request(question, headless=not args.show_browser)

    except SpecialistTaskFailed as error:
        print("\nTASK FAILED")
        print(f"Code:    {error.code}")
        print(f"Details: {error.message}")

        if error.code == "insufficient_context":
            print(
                "\nThe knowledge base does not cover this request. "
                "No ticket was created."
            )

        elif error.code == "unsupported_category":
            print(
                "\nThe support form cannot accept this category. "
                "No ticket was created."
            )

    except TimeoutError as error:
        print("\nTIMEOUT")
        print(error)
        print("\nThe Specialist Agent did not respond in time. "
              "No ticket was created.")

    except ConnectionError as error:
        print("\nCONNECTION ERROR")
        print(error)
        print("\nIs the Specialist Agent running? "
              "Start it with:\n"
              "  python -m uvicorn specialist_agent.server:app")

    except ImportError:
        print("\nPlaywright workflow is not implemented yet "
              "(requester_agent/browser_agent.py).")

    except Exception as error:
        print("\nUNEXPECTED ERROR")
        print(error)


if __name__ == "__main__":
    main()