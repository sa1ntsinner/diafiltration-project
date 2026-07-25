"""
Interactive dashboard for the diafiltration benchmark.

Run with::

    streamlit run src/dfp/dashboard/app.py

The dashboard is a *thin* layer: every number and every figure comes from the
same functions that produce the report (:mod:`dfp.experiments`,
:mod:`dfp.viz`), so the interactive results can never drift away from the
written ones.  All plots are Matplotlib, so the dashboard adds no plotting
dependency beyond the packages allowed by the task sheet.
"""

from __future__ import annotations

import sys
from pathlib import Path

# make `dfp` importable when the file is launched directly by Streamlit
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from dfp.dashboard.views import PAGES, inject_css, sidebar_header  # noqa: E402

st.set_page_config(page_title="Diafiltration NMPC · TU Dortmund",
                   page_icon="🧪", layout="wide",
                   initial_sidebar_state="expanded")
inject_css()
choice = sidebar_header(list(PAGES))
PAGES[choice]()
