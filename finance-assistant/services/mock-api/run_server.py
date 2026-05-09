"""Run mock banking API in-process (avoids Windows uvicorn CLI subprocess loading the wrong app)."""

from __future__ import annotations

import uvicorn

from mock_app import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info", access_log=True)
