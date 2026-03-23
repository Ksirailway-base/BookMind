from pathlib import Path
import pickle
import random
import re

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from llm_client import get_embeddings, get_llm, LOCAL_BASE_URL
from prompt_builder import QUESTION_PROMPT, HOMEWORK_PROMPT, GENERATE_TASK_PROMPT
from task_generator import parse_generated_task, extract_and_format_task
from pdf_parser import parse_pdf

CHROMA_DIR = Path(__file__).parent / "chroma_db"
BOOKS_DIR = Path(__file__).parent / "books"
MAX_TASK_RETRIES = 5

TASK_QUERIES = [
    "grammar vocabulary exercise",
    "put words in correct order",
    "choose the correct form",
    "rewrite the sentences",
    "complete the sentences verb form",
]

def _collection_name(pdf_name: str) -> str:
    name = Path(pdf_name).name
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

    chunks = parse_pdf(path)

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

def _format_context(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        book = doc.metadata.get("book", "unknown")
        chapter = doc.metadata.get("chapter", "")
        section = doc.metadata.get("section", "")
        ctype = doc.metadata.get("content_type", "unknown").upper()
        
        if book.lower().endswith(".pdf"):
            book = book[:-4]
            
        header = f"[{book}, p.{page}] TYPE: {ctype}"
        if chapter:
            header += f" | {chapter}"
        if section:
            header += f" > {section}"
            
        parts.append(f"{header}\n{doc.page_content}")
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

def _build_retriever(collection: str, chroma_filter: dict | None = None):
    search_kwargs: dict = {"k": 3}
    if chroma_filter:
        search_kwargs["filter"] = chroma_filter

    vectorstore = Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    retriever_chroma = vectorstore.as_retriever(search_kwargs=search_kwargs)

    if chroma_filter:
        return retriever_chroma, vectorstore

    bm25_path = CHROMA_DIR / f"{collection}_bm25.pkl"
    if bm25_path.exists():
        with open(bm25_path, "rb") as f:
            bm25_retriever = pickle.load(f)
        bm25_retriever.k = 2
        ensemble = EnsembleRetriever(
            retrievers=[retriever_chroma, bm25_retriever], weights=[0.6, 0.4]
        )
        return ensemble, vectorstore

    return retriever_chroma, vectorstore

def _enrich_with_related_rules(
    docs: list[Document], vectorstore: Chroma
) -> list[Document]:
    """Parent-Child / Cross-type linking: pull related rules for examples."""
    enriched = list(docs)
    added_sections = set(d.metadata.get("section", "") for d in enriched)

    for doc in docs:
        related = doc.metadata.get("related_rule")
        if related and related not in added_sections:
            try:
                                                      
                rule_docs = vectorstore.similarity_search(
                    query="rule",
                    k=2,
                    filter={
                        "$and": [
                            {"section": related},
                            {"content_type": "rule"}
                        ]
                    }
                )
                for r in rule_docs:
                    if r.metadata.get("section") not in added_sections:
                        enriched.append(r)
                        added_sections.add(r.metadata.get("section", ""))
            except Exception:
                pass

    return enriched

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

def _contextualize_query(query: str, history: str, active_task: str, llm) -> str:
    keywords = ["this", "that", "these", "those", "it", "they", "rule", "problem", "exercise"]
    if not any(k in query.lower() for k in keywords):
        return query
        
    if "no previous conversation" in history.lower() and "no active task" in active_task.lower():
        return query

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    prompt = ChatPromptTemplate.from_template(
        "Rewrite the user's question into a standalone search query for a grammar textbook.\n"
        "Resolve words like 'this', 'that', 'these rules', 'these problems' by substituting the actual grammar topic discussed in the history or active task.\n"
        "Return ONLY the rewritten query text, with no quotes, explanations, or intro.\n\n"
        "Active Task:\n{active_task}\n\n"
        "History:\n{history}\n\n"
        "User Question: {query}\n\n"
        "Standalone Query:"
    )
    try:
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"active_task": active_task, "history": history, "query": query}).strip()
    except Exception:
        return query

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
    llm = get_llm(provider=provider, api_key=api_key, model=model)

    chroma_filter = None
    if mode == "generate_task":
        chroma_filter = {"content_type": "exercise"}
    elif mode == "question":
                                                              
        chroma_filter = {"content_type": {"$in": ["rule", "reference", "example", "vocabulary"]}}
    
    retriever, vectorstore = _build_retriever(collection, chroma_filter)

    history_str = "No previous conversation."
    if chat_history:
        recent_history = chat_history[-24:]
        formatted = []
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        history_str = "\n".join(formatted)

    active_task_str = _format_active_task_context(active_task)

    search_question = _contextualize_query(question, history_str, active_task_str, llm)

    if mode == "generate_task":
        seen_pages = set()
        queries = TASK_QUERIES.copy()
        random.shuffle(queries)

        for attempt in range(MAX_TASK_RETRIES):
            search_query = f"{search_question} {queries[attempt % len(queries)]}"
            all_docs = retriever.invoke(search_query)

            fresh_docs = [
                d for d in all_docs
                if d.metadata.get("page") not in seen_pages
            ]

            if not fresh_docs:
                continue

            fresh_docs = _enrich_with_related_rules(fresh_docs, vectorstore)

            for d in fresh_docs:
                seen_pages.add(d.metadata.get("page"))

            for doc in fresh_docs:
                page = str(doc.metadata.get("page", "?"))
                book_id = doc.metadata.get("book", "unknown")
                if book_id.lower().endswith(".pdf"):
                    book_id = book_id[:-4]

                result = extract_and_format_task(doc.page_content, book_id, page, llm=llm)
                if result and result != "NO_TASK_FOUND":
                    new_active_task = _build_active_task_context(result, page, book_id)
                    return result, new_active_task

        return "No suitable exercise found. Try asking a grammar question instead.", active_task

    docs = retriever.invoke(search_question)

    if not docs:
        return "No relevant information found in the book for this query.", active_task

    docs = _enrich_with_related_rules(docs, vectorstore)
    
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