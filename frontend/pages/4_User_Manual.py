"""
User Manual Page — Renders the user manual from docs/user_manual.md.
"""

from pathlib import Path
import streamlit as st
from frontend.common import render_header, render_sidebar, setup_page

setup_page("User Manual", icon="📖")
render_sidebar()
render_header("📖 User Manual",
              "Complete guide for using the Predictive Maintenance System")

# Look for the markdown file in multiple candidate paths
candidates = [
    Path("docs/user_manual.md"),
    Path("/app/docs/user_manual.md"),
    Path(__file__).resolve().parent.parent.parent / "docs" / "user_manual.md",
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
