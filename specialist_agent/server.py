import os
import time

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from specialist_agent.rag_pipeline import get_pipeline
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


# Categories the mock support form's dropdown can actually accept.
# The knowledge base also declares Email and Security, which the
# form does not support. Those produce unsupported_category.
FORM_CATEGORIES = {
    "Account Access",
    "Hardware",
    "Software",
    "Network",
}

# Below this cosine similarity the knowledge base does not
# meaningfully cover the question.
CONFIDENCE_THRESHOLD = float(
    os.getenv("CONFIDENCE_THRESHOLD", "0.35")
)

# Used only to demonstrate the Requester Agent's timeout handling.
SIMULATED_DELAY = float(
    os.getenv("SIMULATED_DELAY_SECONDS", "0")
)


class TaskRequest(BaseModel):
    question: str


def fail(task_id, code, message):
    update_task(
        task_id,
        status="failed",
        error={
            "code": code,
            "message": message,
        },
    )


def process_task(task_id: str, question: str):

    try:
        update_task(task_id, status="working")

        if SIMULATED_DELAY:
            time.sleep(SIMULATED_DELAY)

        result = get_pipeline().answer(question)

        confidence = result.get("confidence", 0.0)

        if confidence < CONFIDENCE_THRESHOLD:
            fail(
                task_id,
                "insufficient_context",
                "No knowledge base passage scored above the "
                f"relevance threshold (best score {confidence}).",
            )
            return

        category = result.get("category", "")

        if category not in FORM_CATEGORIES:
            fail(
                task_id,
                "unsupported_category",
                f"Category '{category}' is valid in the knowledge "
                "base but the support form cannot accept it.",
            )
            return

        update_task(
            task_id,
            status="completed",
            result=result,
        )

    except Exception as error:
        fail(task_id, "internal_error", str(error))


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