import time

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from specialist_agent.task_store import (
    create_task,
    get_task,
    update_task,
)


app = FastAPI(
    title="Specialist Agent A2A API",
    description="A2A-style API used by the Requester Agent",
    version="1.0.0",
)


class TaskRequest(BaseModel):
    question: str


def process_task(task_id: str, question: str):
    try:
        update_task(task_id, status="working")

        time.sleep(2)

        result = {
            "category": "Password Reset",
            "resolution_notes": (
                "Verify the user's identity and provide "
                "password reset instructions."
            ),
        }

        update_task(
            task_id,
            status="completed",
            result=result,
        )

    except Exception as error:
        update_task(
            task_id,
            status="failed",
            error=str(error),
        )


@app.get("/")
def root():
    return {
        "message": "Specialist Agent A2A server is running"
    }


@app.post("/tasks")
def submit_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks,
):
    task = create_task(request.question)

    background_tasks.add_task(
        process_task,
        task["task_id"],
        request.question,
    )

    return {
        "task_id": task["task_id"],
        "status": task["status"],
    }


@app.get("/tasks/{task_id}")
def retrieve_task(task_id: str):
    task = get_task(task_id)

    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Task not found",
        )

    return task