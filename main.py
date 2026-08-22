#!/usr/bin/env python3
"""Run the ccexchange crypto bot directly from the project root."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_VENV = PROJECT_ROOT / ".venv"

# `python3 main.py` should use the same environment as the supervisor. Re-exec
# before importing third-party packages when a local virtual environment exists.
if PROJECT_VENV.is_dir() and Path(sys.prefix) != PROJECT_VENV:
    project_python = PROJECT_VENV / "bin" / "python"
    if project_python.is_file():
        os.execv(str(project_python), [str(project_python), *sys.argv])

# Permit `python main.py` before an editable install while keeping src layout.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ccexchange.runtime import main

if __name__ == "__main__":
    main()
