import gradio as gr

class DocQA:
    def __init__(self) -> None:
        self.val = []

def user(user_message, history):
    return "", history + [[user_message, None]]

def bot(history):
    bot_message = "empty"
    history[-1][1] = bot_message
    return history

def upload_file(doc:DocQA, file_obj):
    doc.val.append(file_obj.name)
    print(doc.val)

with gr.Blocks() as demo:
    doc_qa = gr.State(DocQA())
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot([], elem_id="chatbot").style(height=900)
            msg = gr.Textbox()
            clear = gr.Button("Clear")
            
            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
            clear.click(lambda: None, None, chatbot, queue=False)
        with gr.Column(scale=1):
            with gr.Tab("upload"):
                file = gr.File(
                    label="file",
                    file_types=['.txt', '.md', '.docx', '.pdf']
                )
            load_file_button = gr.Button("Load file")

    load_file_button.click(upload_file, inputs=[doc_qa, file])

if __name__ == "__main__":
    demo.launch()