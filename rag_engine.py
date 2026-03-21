from pathlib import Path
import pickle
import random
import re

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from llm_client import get_embeddings, get_llm, LOCAL_BASE_URL
from prompt_builder import QUESTION_PROMPT, HOMEWORK_PROMPT, GENERATE_TASK_PROMPT
from task_generator import parse_generated_task, extract_and_format_task

CHROMA_DIR = Path(__file__).parent / "chroma_db"
BOOKS_DIR = Path(__file__).parent / "books"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MAX_TASK_RETRIES = 5

TASK_QUERIES = [
    "grammar vocabulary exercise fill in the blanks complete the sentences task",
    "put words in correct order exercise practice",
    "choose the correct form complete the gap",
    "rewrite the sentences using the correct tense",
    "complete the sentences verb form exercise",
]


def _collection_name(pdf_name: str) -> str:
    name = pdf_name
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)
    safe = safe.strip("_-")
    if len(safe) < 3:
        safe = safe + "___"
    return safe[:63]


def ingest_pdf(pdf_path: str) -> str:
    path = Path(pdf_path)
    collection = _collection_name(path.name)

    loader = PyMuPDFLoader(str(path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        chunk.metadata["book"] = path.name
        if "page" in chunk.metadata:
            chunk.metadata["page"] = chunk.metadata["page"] + 1

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=collection,
        persist_directory=str(CHROMA_DIR),
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_path = CHROMA_DIR / f"{collection}_bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)

    return collection


def _format_context(docs) -> str:
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        book = doc.metadata.get("book", "unknown")
        if book.lower().endswith(".pdf"):
            book = book[:-4]
        parts.append(f"[{book}, p.{page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def list_books() -> list[str]:
    books = []
    if BOOKS_DIR.exists():
        books = [f.stem for f in BOOKS_DIR.glob("*.pdf")]
    return sorted(set(books))


def is_book_ingested(book_name: str) -> bool:
    if not CHROMA_DIR.exists():
        return False
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collections = [c.name for c in client.list_collections()]
        return _collection_name(book_name) in collections
    except Exception:
        return False


def _build_retriever(collection: str):
    vectorstore = Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    retriever_chroma = vectorstore.as_retriever(search_kwargs={"k": 2})

    bm25_path = CHROMA_DIR / f"{collection}_bm25.pkl"
    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            bm25_retriever = pickle.load(f)
        bm25_retriever.k = 2
        return EnsembleRetriever(
            retrievers=[retriever_chroma, bm25_retriever], weights=[0.6, 0.4]
        )
    return retriever_chroma


def _extract_rule_context(raw_answer: str) -> str:
    match = re.search(r'RULE_CONTEXT:\s*(.*?)(?=\nSENTENCES:|\Z)', raw_answer, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _build_active_task_context(task_text: str, page: str, book_id: str, exercise: str = "?") -> dict:
    return {
        "unit": "?",
        "exercise": exercise,
        "rule_context": "",
        "task_text": task_text,
        "page": page,
        "book": book_id,
    }


def _format_active_task_context(active_task: dict | None) -> str:
    if not active_task:
        return "No active task."
    return (
        f"Unit {active_task.get('unit', '?')}, "
        f"Exercise {active_task.get('exercise', '?')} "
        f"(page {active_task.get('page', '?')})\n"
        f"Grammar rule: {active_task.get('rule_context', 'not specified')}\n"
        f"Task: {active_task.get('task_text', '')}"
    )


def ask(
    question: str,
    book_name: str,
    mode: str = "question",
    provider: str = "local",
    api_key: str = "",
    model: str = "",
    chat_history: list | None = None,
    active_task: dict | None = None,
) -> tuple[str, dict | None]:
    collection = _collection_name(book_name)
    retriever = _build_retriever(collection)
    llm = get_llm(provider=provider, api_key=api_key, model=model)

    history_str = "No previous conversation."
    if chat_history:
        recent_history = chat_history[-24:]
        formatted = []
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        history_str = "\n".join(formatted)

    active_task_str = _format_active_task_context(active_task)

    if mode == "generate_task":
        seen_pages = set()
        queries = TASK_QUERIES.copy()
        random.shuffle(queries)

        for attempt in range(MAX_TASK_RETRIES):
            search_query = queries[attempt % len(queries)]
            all_docs = retriever.invoke(search_query)

            fresh_docs = [
                d for d in all_docs
                if d.metadata.get("page") not in seen_pages
            ]

            if not fresh_docs:
                continue

            for d in fresh_docs:
                seen_pages.add(d.metadata.get("page"))

            for doc in fresh_docs:
                page = str(doc.metadata.get("page", "?"))
                book_id = doc.metadata.get("book", "unknown")
                if book_id.lower().endswith(".pdf"):
                    book_id = book_id[:-4]

                result = extract_and_format_task(doc.page_content, book_id, page)
                if result:
                    new_active_task = _build_active_task_context(result, page, book_id)
                    return result, new_active_task

        return "No suitable exercise found. Try asking a grammar question instead.", active_task

    docs = retriever.invoke(question)

    if not docs:
        return "No relevant information found in the book for this query.", active_task

    context = _format_context(docs)

    if mode == "homework":
        chain = HOMEWORK_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "question": question,
            "active_task_context": active_task_str,
        })
        return answer, active_task

    chain = QUESTION_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "chat_history": history_str,
        "question": question,
        "active_task_context": active_task_str,
    })
    return answer, active_task