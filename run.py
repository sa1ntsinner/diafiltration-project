#!/usr/bin/env python3
"""
Entry point that works from a fresh clone, without installing anything.

    python run.py optimum --numeric
    python run.py bench
    python run.py all
    python run.py run tear objectives
    python run.py list

It only puts ``src/`` on ``sys.path`` and hands over to :mod:`dfp.cli`.  After
``pip install -e .`` the same commands are available as ``python -m dfp.cli …``
or simply ``dfp …``.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dfp.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
