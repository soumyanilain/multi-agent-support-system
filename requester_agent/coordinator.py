import json
import os

from requester_agent.a2a_client import A2AClient


# Configuration
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


def run_request(question):
    """
    Main Requester Agent workflow.

    1. Receives the user's question
    2. Submits it to the Specialist Agent
    3. Receives a task ID
    4. Polls until completed/failed/timeout
    5. Returns the Specialist Agent result
    """

    client = A2AClient(base_url=SPECIALIST_URL)

    print("\n--- Requester Agent ---")
    print(f"User request: {question}")

    # Step 1: Submit task
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

    # Step 2: Poll for result
    print("\nWaiting for Specialist Agent...")

    result = client.wait_for_result(
        task_id=task_id,
        timeout=TASK_TIMEOUT,
        poll_interval=POLL_INTERVAL
    )

    # Step 3: Display the completed result
    print("\nTask completed!")
    print("Specialist result:")

    print(
        json.dumps(
            result,
            indent=4
        )
    )

    return result


def main():
    print("==============================")
    print(" Multi-Agent Support System")
    print("==============================")

    question = input(
        "\nEnter the customer's issue: "
    ).strip()

    if not question:
        print("Error: A question is required.")
        return

    try:
        result = run_request(question)

        # Later Person C's Playwright function can be called here.
        #
        # Example:
        #
        # from requester_agent.browser_agent import submit_ticket
        # success = submit_ticket(result)
        #
        # if success:
        #     print("Support ticket submitted successfully.")

    except TimeoutError as error:
        print("\nTIMEOUT:")
        print(error)

    except ConnectionError as error:
        print("\nCONNECTION ERROR:")
        print(error)

    except RuntimeError as error:
        print("\nTASK FAILED:")
        print(error)

    except Exception as error:
        print("\nUNEXPECTED ERROR:")
        print(error)


if __name__ == "__main__":
    main()