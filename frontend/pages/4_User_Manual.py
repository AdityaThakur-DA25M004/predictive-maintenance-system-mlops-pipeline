"""
User Manual Page — Renders the user manual from docs/user_manual.md.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path
import streamlit as st
from frontend.common import render_header, render_sidebar, setup_page

setup_page("User Manual", icon="📖")
render_sidebar()
render_header("📖 User Manual",
              "Complete guide for using the Predictive Maintenance System")

# Look for the markdown file in multiple candidate paths
_here = Path(__file__).resolve()
candidates = [
    # Running locally from project root
    Path("user_manual.md"),
    Path("docs/user_manual.md"),
    # Relative to this page file (frontend/pages/ → project root)
    _here.parent.parent / "user_manual.md",
    _here.parent.parent / "docs" / "user_manual.md",
    # Docker container paths
    Path("/app/user_manual.md"),
    Path("/app/docs/user_manual.md"),
]

content = None
for p in candidates:
    if p.exists():
        content = p.read_text(encoding="utf-8")
        break

if content:
    st.markdown(content)
else:
    st.warning("`docs/user_manual.md` not found. Ship it with the repo.")