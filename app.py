import os
import re
import shutil
import chromadb
import docx
import openai
import json
import gradio as gr
import requests
from chromadb.config import Settings
from pdfminer.high_level import extract_text
from langchain.text_splitter import CharacterTextSplitter

LLM_MODEL = "chatglm-6b"

def pdf2text(path):
    return extract_text(path)

def txt2text(path):
    with open(path, "r") as f:
        return f.read()

def md2text(path):
    return txt2text(path)

def doc2text(path):
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])


chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="chromadb_collection"
        ))

def query_glm(query, system_context):
    url = "http://127.0.0.1:3000/gpt"

    headers = {
    'Content-Type': 'application/json'
    }
    payload = json.dumps({
        "query": f"{system_context}\n{query}"
    })
    completion = requests.post(url=url,headers=headers, data=payload)
    if completion.status_code != 200:
        return "error"
    return json.loads(completion.content)["response"]


def query_openai(query, system_context):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_context},
            {"role": "user", "content": query}
        ]
    )
    return completion.choices[0].message.content

class DocQA:
    def __init__(self) -> None:
        self.val = []
        self.collection = None
        self.text_splitter = CharacterTextSplitter(chunk_size=1000)
    
    def load_document(self, path):
        collection_name = re.sub(r'\W+', '', os.path.basename(path)).lower()
        self.collection = chroma_client.get_or_create_collection(collection_name)
        if self.collection.count() == 0:
            text = self.file2text(path)
            documents = self.text_splitter.split_text(text)
            self.collection.add(
                ids = [str(i) for i in range(len(documents))],
                documents=documents,
            )
    
    def query(self, query, n_results=2, model="gpt-3.5-turbo"):
        query_model = {
            "gpt-3.5-turbo": query_openai,
            "chatglm-6b": query_glm
        }
        if self.collection is None:
            return "No context"
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        system_context = "I will ask you a question. You must answer the question based on the context."
        prompt_template = "Here is the context: \n{context}\n. This is my question: {query}"
        
        context = '\n'.join(results['documents'][0])
        res = query_model.get(model, query_openai)(prompt_template.format(context=context, query=query), system_context)
        return res
    
    @staticmethod
    def file2text(path):
        text = None
        if path.endswith(".pdf"):
            text = pdf2text(path)
        elif path.endswith(".md"):
            text = md2text(path)
        elif path.endswith(".txt"):
            text = txt2text(path)
        else:
            text = doc2text(path)
        return text

def user(user_message, history, ):
    # res = doc.query(user_message, n_results=1)
    return "", history + [[user_message, None]]

def bot(history, doc:DocQA, model):
    bot_message = doc.query(history[-1][0], n_results=1, model=model)
    history[-1][1] = bot_message
    return history

def upload_file(doc:DocQA, file_obj, model):
    doc.load_document(file_obj.name)
    return doc.query("summarize the document", n_results=3, model=model)

def upload_file_fn(f):
    if not os.path.exists("content"):
        os.mkdir("content")
    filename = os.path.basename(f.name)
    tgt_path = os.path.join("content", filename)
    if not os.path.exists(tgt_path):
        shutil.move(f.name, tgt_path)
    

with gr.Blocks() as demo:
    doc_qa = gr.State(DocQA())
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot([], elem_id="chatbot").style(height=800)
            msg = gr.Textbox()
            clear = gr.Button("Clear")
        
        with gr.Column(scale=2):
            llm_model = gr.Radio(
                ["chatglm-6b", "gpt-3.5-turbo"], 
                elem_id="llm_model", 
                value=LLM_MODEL,
                label='LLM Model',
                interactive=True)
            with gr.Tab("Upload"):
                file = gr.File(
                    label="Support file type (txt, md, doc, pdf)",
                    file_types=['.txt', '.md', '.docx', '.doc', '.pdf']
                )
                load_file_button = gr.Button("Load file")
            
            summary = gr.TextArea(
                label="Summary", 
                elem_id="output", 
                interactive=False,
            ).style(show_copy_button=True)

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, [chatbot, doc_qa, llm_model], chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)
    file.upload(upload_file_fn, inputs=file)
    load_file_button.click(upload_file, inputs=[doc_qa, file, llm_model], outputs=summary)


if __name__ == "__main__":
    demo.launch()