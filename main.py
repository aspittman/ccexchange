#!/usr/bin/env python3
"""Run the ccexchange crypto bot directly from the project root."""

import sys
from pathlib import Path

# Permit `python main.py` before an editable install while keeping src layout.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ccexchange.runtime import main

if __name__ == "__main__":
    main()
