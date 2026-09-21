import uuid


# In-memory dictionary used to store all tasks
tasks = {}


def create_task(question):
    """
    Create a new task and give it a unique ID.
    """

    task_id = str(uuid.uuid4())

    tasks[task_id] = {
        "task_id": task_id,
        "question": question,
        "status": "submitted",
        "result": None,
        "error": None
    }

    return tasks[task_id]


def get_task(task_id):
    """
    Retrieve a task by its ID.
    """

    return tasks.get(task_id)


def update_task(task_id, status=None, result=None, error=None):
    """
    Update an existing task.
    """

    task = tasks.get(task_id)

    if task is None:
        return None

    if status is not None:
        task["status"] = status

    if result is not None:
        task["result"] = result

    if error is not None:
        task["error"] = error

    return task