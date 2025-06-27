import gradio as gr
from model_setup import retriever, qa_pipeline  

def ask_flan_gradio_chat(question, chat_history):
    top_k = 1
    max_tokens = 500

    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs[:top_k]])

    prompt = f"""You are a medical assistant specialized in neurology and psychiatry.
Answer the question based on the context.

Context:
{context}

Question:
{question}"""

    result = qa_pipeline(prompt, max_new_tokens=max_tokens)
    answer = result[0]["generated_text"].strip()

    chat_history = chat_history or []
    chat_history.append((question, answer))
    return chat_history, chat_history

with gr.Blocks(css="""
    body {background-color: #47878E;}
    .gradio-container {
        background-color: #47878E !important;
        min-height: 100vh;
        padding: 2rem;
    }
    .chatbot {
        border-radius: 12px;
        border: 1px solid #253854;
        background-color: #F0F2F2;
    }
    .gr-button {
        background-color: #253854 !important;
        color: white !important;
    }
""") as demo:

    gr.Markdown("<h1 style='text-align:center; color:#FFFFFF;'>🧠 NeuroAssistant</h1>")

    chatbot = gr.Chatbot(elem_classes="chatbot")
    msg = gr.Textbox(placeholder="Type your question here...", label="Your Question")
    submit = gr.Button("Ask")

    submit.click(
        ask_flan_gradio_chat,
        inputs=[msg, chatbot],
        outputs=[chatbot, chatbot]
    )

    msg.submit(
        ask_flan_gradio_chat,
        inputs=[msg, chatbot],
        outputs=[chatbot, chatbot]
    )

demo.launch(share=True)