# Group Project 1: Multi-Agent AI System

**ITCS 5010 — Group 6**
Soumyanil Ain · Sogol Maghzian · Meghana Thummalapally · Sumiran Juthuga

A two-agent system that combines Agent-to-Agent (A2A) communication,
Retrieval-Augmented Generation (RAG), and browser automation to file a
support ticket end to end.

```
User request
    -> Requester Agent      (coordinator.py)
    -> A2A submit + poll    (a2a_client.py  <->  server.py)
    -> Specialist Agent     (rag_pipeline.py)
    -> RAG + Groq LLM
    -> grounded result
    -> Playwright           (browser_agent.py)
    -> ticket confirmed
```

---

## 1. One-time setup

### 1.1 Clone and enter the project

```bash
git clone https://github.com/soumyanilain/multi-agent-support-system.git
cd multi-agent-support-system
```

### 1.2 Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

Your prompt should now start with `(venv)`. **Every new terminal needs this
activation step.**

### 1.3 Install dependencies

```bash
pip install -r requirements.txt
playwright install
```

The first install takes 5–10 minutes — `sentence-transformers` pulls in
PyTorch.

### 1.4 Get a Groq API key

1. Go to https://console.groq.com
2. Sign in, then **API Keys → Create API Key**
3. Copy it immediately (it is shown only once)

It is free.

### 1.5 Create your `.env`

Copy the template:

**Windows:** `Copy-Item .env.example .env`
**macOS / Linux:** `cp .env.example .env`

Then open `.env` and set two values:

```
GROQ_API_KEY=gsk_your_actual_key_here
MOCK_APP_PATH=D:/your/path/to/multi-agent-support-system/mock_support_app/index.html
```

Use forward slashes in `MOCK_APP_PATH`, even on Windows.

> **Never commit `.env`.** It is gitignored, and the repository has GitHub
> push protection enabled. If a key is ever committed, revoke it at
> console.groq.com immediately rather than trying to delete the commit.

---

## 2. Running the system

You need **two terminals**, both with the virtual environment activated.

### Terminal 1 — start the Specialist Agent

```bash
python -m uvicorn specialist_agent.server:app --reload
```

Wait for `Application startup complete`. Leave this running.

The RAG pipeline builds lazily, so the model loading messages appear on the
first request, not at startup.

### Terminal 2 — run the Requester Agent

```bash
python -m requester_agent.coordinator --question "I forgot my password and cannot log into my account."
```

Add `--show-browser` to watch Playwright work (used for the demo video):

```bash
python -m requester_agent.coordinator --question "I forgot my password and cannot log into my account." --show-browser
```

Omit `--question` to be prompted interactively.

**Expected output:** task submitted → status `working` → status `completed` →
grounded result with sources → browser fills the form → 5-digit ticket ID.

---

## 3. Running the evaluation

The Specialist Agent does **not** need to be running for this.

```bash
python -m tests.run_eval
```

Produces three tables: classification accuracy, failure-mode detection, and a
dense-only vs hybrid-fusion retrieval comparison. Takes a couple of minutes.

---

## 4. Testing the failure modes

All four are demonstrable without changing any code.

### 4.1 Out-of-scope question (`insufficient_context`)

```bash
python -m requester_agent.coordinator --question "How do I rebuild the transmission in my car?"
```

Retrieval scores 0.104, below the 0.30 threshold. The task fails and **the
browser never opens.**

### 4.2 Category the form cannot accept (`unsupported_category`)

```bash
python -m requester_agent.coordinator --question "My company email is not synchronizing."
```

The RAG correctly classifies this as `Email`, but the support form's dropdown
only offers Account Access, Hardware, Software, and Network. The task fails
before automation begins.

### 4.3 Timeout

In `.env`, set:

```
SIMULATED_DELAY_SECONDS=45
```

Restart Terminal 1 (Ctrl+C, then start uvicorn again), then run any question.
The Requester polls for 30 seconds and reports a timeout.

**Set it back to `0` afterwards.**

### 4.4 Specialist Agent unreachable

Stop the server in Terminal 1 (Ctrl+C), then run any question. The client
retries with exponential backoff, then reports a connection error with
instructions to start the server.

---

## 5. Opening the mock application manually

It is a plain local file with no server. Double-click:

```
mock_support_app/index.html
```

Do **not** try to reach it through `localhost:8000` — that port is the
Specialist Agent's API, which only serves `/`, `/tasks`, and `/tasks/{id}`.

---

## 6. Project structure

```
specialist_agent/
    server.py         FastAPI A2A endpoints, failure policies
    task_store.py     In-memory task registry and status transitions
    rag_pipeline.py   Loading, chunking, embeddings, FAISS, BM25, fusion, LLM
requester_agent/
    coordinator.py    End-to-end workflow and error reporting
    a2a_client.py     Task submission, polling, timeout, backoff
    browser_agent.py  Playwright automation and confirmation verification
knowledge_base/       7 course-provided support documents
mock_support_app/     The provided ticket form (footer added by our team)
tests/
    test_cases.json   5 provided test cases
    run_eval.py       Evaluation harness
docs/
    CONTRACT.md       Frozen JSON schema shared by all components
```

---

## 7. A2A protocol

| Endpoint               | Purpose                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------ |
| `POST /tasks`          | Submit a question. Returns `task_id` **immediately**; RAG runs in a background task. |
| `GET /tasks/{task_id}` | Poll for status and result. 404 if unknown.                                          |

Status lifecycle: `submitted` → `working` → `completed` \| `failed`

Failure codes: `insufficient_context`, `unsupported_category`, `llm_error`,
`internal_error`

---

## 8. Configuration reference

| Variable                  | Default                 | Purpose                                    |
| ------------------------- | ----------------------- | ------------------------------------------ |
| `GROQ_API_KEY`            | —                       | Required                                   |
| `GROQ_MODEL`              | `openai/gpt-oss-20b`    | LLM used for query expansion and answers   |
| `SPECIALIST_URL`          | `http://127.0.0.1:8000` | Where the Requester finds the Specialist   |
| `MOCK_APP_PATH`           | repo copy               | Absolute path to `index.html`              |
| `POLL_INTERVAL_SECONDS`   | `2`                     | Seconds between status polls               |
| `TASK_TIMEOUT_SECONDS`    | `30`                    | Requester gives up after this              |
| `CONFIDENCE_THRESHOLD`    | `0.30`                  | Minimum cosine similarity to answer        |
| `SIMULATED_DELAY_SECONDS` | `0`                     | Set high to demo the timeout               |
| `EVAL_PACE_SECONDS`       | `2`                     | Pause between eval questions (rate limits) |

---

## 9. Common problems

**`uvicorn is not recognized`** — the virtual environment is not activated, or
use `python -m uvicorn` instead.

**`curl` errors in PowerShell** — PowerShell aliases `curl` to
`Invoke-WebRequest`. Use `curl.exe`, or `Invoke-RestMethod`.

**`GROQ_API_KEY was not found`** — `.env` is missing or the key line is blank.

**`429 rate_limit_exceeded`** — Groq's free tier caps tokens per minute. Wait a
minute, or raise `EVAL_PACE_SECONDS`.

**Mock app not found** — check `MOCK_APP_PATH` in `.env` uses an absolute path
with forward slashes.
