from io import BytesIO
import streamlit as st
from streamlit_chat import message
from handle import LlamaSummarize
from pdfminer.high_level import extract_text

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []
    

summary_prompt = st.text_area('Write a prompt to summarize',
                              '''To summarise the paper, the summary needs to include these aspects: 
    1. the background to the paper
    2. the relevant research
    3. the methods used
    4. the result of the paper
Then list a few key points.
Finally give a few possible questions about the paper and provide answers.''', height=300)


uploaded_file = st.file_uploader("Choose a PDF file (only PDF supported!!!)")
if uploaded_file is not None:
    bytesio = BytesIO(uploaded_file.getvalue())
    text = extract_text(bytesio)
    doc_index = LlamaSummarize(text)
    st.session_state["doc_index"] = doc_index


if st.button("Summarize") and uploaded_file is not None:
    summarize = st.session_state["doc_index"].summarize(summary_prompt)
    st.write(summarize)
    st.session_state["button_able"] = False


def get_text():
    input_text = st.text_input(
        "Ask something about the document: ", key="input")
    return input_text


user_input = get_text()


if user_input:
    st.session_state.past.append(user_input)
    st.session_state.generated.append(user_input)


if len(st.session_state.generated) > 0:
    for i in range(len(st.session_state.generated)-1, -1, -1):
        message(st.session_state.generated[i], key=str(
            i), avatar_style="thumbs")
        message(st.session_state.past[i], is_user=True, key=str(i) + '_user')
