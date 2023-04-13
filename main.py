from io import BytesIO
import streamlit as st
from handle import LlamaSummarize
from pdfminer.high_level import extract_text

summary_prompt = st.text_area('Write a prompt to summarize',
                              '''To summarise the paper, the summary needs to include these aspects: 
    1. the background to the paper
    2. the relevant research
    3. the methods used
    4. the result of the paper
Then list a few key points.
Finally give a few possible questions about the paper and provide answers.''',height=300)


uploaded_file = st.file_uploader("Choose a PDF file (only PDF supported!!!)")
if uploaded_file is not None:
    bytesio = BytesIO(uploaded_file.getvalue())
    text = extract_text(bytesio)

if st.button("Summarize"):
    summarize = LlamaSummarize(text).summarize(summary_prompt)
    st.write(summarize)