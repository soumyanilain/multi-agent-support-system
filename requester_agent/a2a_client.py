import time
import requests


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

    def wait_for_result(self, task_id, timeout=30, poll_interval=1):
        """
        Poll the Specialist Agent until the task completes,
        fails, or reaches the timeout.
        """

        start_time = time.time()

        while time.time() - start_time < timeout:

            try:
                task = self.get_task(task_id)

            except ConnectionError as error:
                print(f"Connection error: {error}")
                time.sleep(poll_interval)
                continue

            status = task.get("status")

            print(f"Task {task_id} status: {status}")

            if status == "completed":
                return task.get("result")

            if status == "failed":
                error_message = task.get(
                    "error",
                    "Specialist Agent failed to complete the task."
                )

                raise RuntimeError(error_message)

            time.sleep(poll_interval)

        raise TimeoutError(
            f"Task {task_id} did not finish within {timeout} seconds."
        )