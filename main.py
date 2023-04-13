from io import BytesIO
import streamlit as st
from handle import LlamaSummarize
from pdfminer.high_level import extract_text

uploaded_file = st.file_uploader("Choose a PDF file (only PDF supported!!!)")
if uploaded_file is not None:
    bytesio = BytesIO(uploaded_file.getvalue())
    text = extract_text(bytesio)
    summarize = LlamaSummarize(text).summarize()
    st.write(summarize)
