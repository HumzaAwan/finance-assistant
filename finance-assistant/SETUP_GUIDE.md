# Quick setup

## 1. Install Ollama and models

Install from [ollama.com](https://ollama.com), start the app/service, then in a terminal:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## 2. Python environment

From the `finance-assistant` folder:

```bash
python -m venv .venv
```

Activate:

- **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`
- **Windows (Git Bash) / macOS / Linux:** `source .venv/Scripts/activate` or `source .venv/bin/activate`

```bash
python -m pip install --upgrade pip
pip install -r requirements-local.txt
```

## 3. Env file

Either copy `.env.example` → `.env`, or skip copying — **`python run_local.py` creates `.env` from `.env.example` if it’s missing**.

Edit `.env` only if your ports or URLs differ (defaults assume everything on localhost).

## 4. Run

```bash
python run_local.py
```

First run builds the mock DB and **indexes RAG** (needs Ollama). Next times you can use:

```bash
python run_local.py --skip-ingest
```

Open **http://127.0.0.1:8501**. Use User ID **`user_001`** in the sidebar (matches mock data).

**Stop:** Ctrl+C.

---

**Flags:** `--skip-ingest` (skip RAG re-index), `--no-ui` (APIs only). **Windows port issues:** try `python run_local.py --free-ports`.

More detail: **`README.md`**, **`PROJECT_OVERVIEW.md`**.
