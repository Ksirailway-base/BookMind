import shutil
from pathlib import Path

import gradio as gr
from rag_engine import ingest_pdf, ask, list_books, is_book_ingested, BOOKS_DIR

HOMEWORK_TRIGGERS = [
    "homework", "check", "check my answer", "check this",
    "is this correct", "evaluate", "grade", "review",
]

TASK_TRIGGERS = [
    "give me a homework", "give me homework", "can you give me a homework",
    "can you give me homework", "some homework", "homework please",
    "give me a task", "give me one task", "give me task",
    "give me another task", "generate a task", "create a task",
    "exercise", "give me an exercise", "give me exercise"
]


def detect_mode(message: str) -> str:
    lower = message.lower()
    for trigger in TASK_TRIGGERS:
        if trigger in lower:
            return "generate_task"
    for trigger in HOMEWORK_TRIGGERS:
        if trigger in lower:
            return "homework"
    return "question"


def get_book_choices() -> list[str]:
    pdf_books = []
    if BOOKS_DIR.exists():
        pdf_books = [f.stem for f in BOOKS_DIR.glob("*.pdf")]
    return sorted(set(pdf_books))


def on_upload(file, progress=gr.Progress(track_tqdm=False)):
    if file is None:
        return gr.update(), "⚠️ No file selected."
    filename = Path(file).name
    if not filename.lower().endswith(".pdf"):
        return gr.update(), "⚠️ Upload a PDF file."

    BOOKS_DIR.mkdir(exist_ok=True)
    dest_path = BOOKS_DIR / filename
    shutil.copy2(file, dest_path)

    stem = Path(filename).stem

    if not is_book_ingested(stem):
        progress(0, desc="")
        progress(0.2, desc="Indexing...")
        ingest_pdf(str(dest_path))
        progress(1.0, desc="Done!")

    choices = get_book_choices()
    return gr.update(choices=choices, value=stem), f"✅ {stem} — indexed and ready."


def on_refresh():
    choices = get_book_choices()
    return gr.update(choices=choices, value=choices[0] if choices else None)


def respond(message, chat_history, book_name, provider, api_key, model_name, active_task):
    if not message.strip():
        return "", chat_history, active_task

    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": message})

    if not book_name:
        chat_history.append({"role": "assistant", "content": "⚠️ Select a book first."})
        yield "", chat_history, active_task
        return

    if "OpenAI" in provider:
        prov = "openai"
        if not api_key:
            chat_history.append({"role": "assistant", "content": "⚠️ Enter your OpenAI API key in Settings."})
            yield "", chat_history, active_task
            return
    elif "Gemini" in provider:
        prov = "gemini"
        if not api_key:
            chat_history.append({"role": "assistant", "content": "⚠️ Enter your Gemini API key in Settings."})
            yield "", chat_history, active_task
            return
    else:
        prov = "local"
        api_key = ""

    if not is_book_ingested(book_name):
        chat_history.append({"role": "assistant", "content": "⚠️ This book is not indexed yet. Try re-uploading."})
        yield "", chat_history, active_task
        return

    chat_history.append({"role": "assistant", "content": "🤔 Thinking..."})
    yield "", chat_history, active_task

    mode = detect_mode(message)

    try:
        answer, new_active_task = ask(
            question=message,
            book_name=book_name,
            mode=mode,
            provider=prov,
            api_key=api_key,
            model=model_name or "",
            chat_history=chat_history[:-1],
            active_task=active_task,
        )
    except Exception as e:
        answer = f"❌ Error: {e}"
        new_active_task = active_task

    chat_history[-1] = {"role": "assistant", "content": answer}
    yield "", chat_history, new_active_task


THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
)

CSS = """
.gradio-container {
    max-width: 100% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}
h1 {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2em !important;
    margin-bottom: 0 !important;
}
.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 0.9em;
    margin-top: 0 !important;
    margin-bottom: 0.5em;
}
input[type="radio"]:checked + span {
    font-weight: bold;
}
.scrollable-list {
    max-height: 480px;
    overflow-y: auto;
}
.radio-container {
    padding-top: 20px;
}
"""

with gr.Blocks(title="BookMind") as app:
    gr.Markdown("# BookMind")
    gr.Markdown('<p class="subtitle">Local AI teacher for any textbook</p>')

    active_task_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            with gr.Accordion("⚙️ Settings", open=False):
                provider_radio = gr.Radio(
                    choices=["Local", "OpenAI", "Gemini"],
                    value="Local",
                    label="LLM Provider",
                    info="Local = llama-server",
                )
                api_key_box = gr.Textbox(label="API Key", placeholder="sk-... or AIza...", type="password", visible=False)
                model_name_box = gr.Textbox(
                    label="Model",
                    placeholder="e.g. gpt-4o-mini, gemini-2.5-flash-lite",
                    visible=False,
                )

                def on_provider_change(provider):
                    is_cloud = provider != "Local"
                    default_model = ""
                    if provider == "OpenAI":
                        default_model = "gpt-4o-mini"
                    elif provider == "Gemini":
                        default_model = "gemini-2.5-flash-lite"
                    return gr.update(visible=is_cloud), gr.update(visible=is_cloud, value=default_model)

                provider_radio.change(fn=on_provider_change, inputs=provider_radio, outputs=[api_key_box, model_name_box])

            with gr.Row():
                upload_btn = gr.UploadButton("📤 Upload", file_types=[".pdf"], scale=1)
                refresh_btn = gr.Button("🔄 Refresh", scale=1)

            status_bar = gr.Markdown("")
            with gr.Column(elem_classes="scrollable-list"):
                book_radio = gr.Radio(choices=get_book_choices(), label="Library (Select a Book)", interactive=True, elem_classes="radio-container")

        with gr.Column(scale=4, min_width=600):
            chatbot = gr.Chatbot(height=650, placeholder="Select a book, then ask away!")
            msg = gr.Textbox(placeholder="Ask a question...", lines=1, show_label=False)

            with gr.Row():
                send_btn = gr.Button("Send", variant="primary", scale=0, min_width=120)

    upload_btn.upload(fn=on_upload, inputs=upload_btn, outputs=[book_radio, status_bar])
    refresh_btn.click(fn=on_refresh, outputs=book_radio)

    send_btn.click(
        fn=respond,
        inputs=[msg, chatbot, book_radio, provider_radio, api_key_box, model_name_box, active_task_state],
        outputs=[msg, chatbot, active_task_state],
    )
    msg.submit(
        fn=respond,
        inputs=[msg, chatbot, book_radio, provider_radio, api_key_box, model_name_box, active_task_state],
        outputs=[msg, chatbot, active_task_state],
    )

    gr.Markdown('<center><small>BookMind • LangChain · Chroma · llama.cpp · Gradio 6</small></center>')

if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        share=False,
        inbrowser=True,
        theme=THEME,
        css=CSS,
    )