import time

import requests


class SpecialistTaskFailed(RuntimeError):
    """
    Raised when the Specialist Agent reports status "failed".

    Carries the structured error code so the Requester Agent can
    react differently to each failure mode.
    """

    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


class A2AClient:
    """
    Client used by the Requester Agent to communicate
    with the Specialist Agent.
    """

    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url

    def submit_task(self, question):
        """
        Submit a new task to the Specialist Agent.

        Returns:
            {
                "task_id": "...",
                "status": "submitted"
            }
        """

        try:
            response = requests.post(
                f"{self.base_url}/tasks",
                json={"question": question},
                timeout=5
            )

            response.raise_for_status()

            return response.json()

        except requests.RequestException as error:
            raise ConnectionError(
                f"Could not submit task to Specialist Agent: {error}"
            )

    def get_task(self, task_id):
        """
        Get the current status/result of a task.
        """

        try:
            response = requests.get(
                f"{self.base_url}/tasks/{task_id}",
                timeout=5
            )

            response.raise_for_status()

            return response.json()

        except requests.RequestException as error:
            raise ConnectionError(
                f"Could not retrieve task {task_id}: {error}"
            )

    def wait_for_result(self, task_id, timeout=30, poll_interval=2):
        """
        Poll the Specialist Agent until the task completes,
        fails, or reaches the timeout.

        Raises:
            SpecialistTaskFailed: the Specialist reported "failed".
            TimeoutError: the task did not finish in time.
            ConnectionError: the Specialist stayed unreachable.
        """

        start_time = time.time()

        consecutive_errors = 0
        max_consecutive_errors = 3

        while time.time() - start_time < timeout:

            try:
                task = self.get_task(task_id)
                consecutive_errors = 0

            except ConnectionError as error:
                consecutive_errors += 1

                print(
                    f"Connection error "
                    f"({consecutive_errors}/{max_consecutive_errors}): "
                    f"{error}"
                )

                if consecutive_errors >= max_consecutive_errors:
                    raise ConnectionError(
                        "Specialist Agent unreachable after "
                        f"{max_consecutive_errors} attempts."
                    )

                # Exponential backoff: 1s, 2s, 4s
                time.sleep(2 ** (consecutive_errors - 1))
                continue

            status = task.get("status")

            print(f"Task {task_id} status: {status}")

            if status == "completed":
                return task.get("result")

            if status == "failed":
                error = task.get("error") or {}

                if isinstance(error, dict):
                    raise SpecialistTaskFailed(
                        error.get("code", "unknown_error"),
                        error.get(
                            "message",
                            "Specialist Agent failed to complete the task."
                        )
                    )

                raise SpecialistTaskFailed("unknown_error", str(error))

            time.sleep(poll_interval)

        raise TimeoutError(
            f"Task {task_id} did not finish within {timeout} seconds."
        )