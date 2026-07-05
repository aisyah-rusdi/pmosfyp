import streamlit as st
import os

def load_css(path="assets/style.css"):
    # Resolve relative to the src/ folder (two levels up from utils/ui_helpers.py),
    # so it works no matter what directory Streamlit's working directory is.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, path)
    with open(full_path, encoding='utf-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)