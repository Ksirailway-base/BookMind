from pathlib import Path
import pickle
import random
import re

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from llm_client import get_embeddings, get_llm, get_reranker, LOCAL_BASE_URL
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

def _rerank_documents(query: str, documents: list, top_n: int = 5):
    if not documents: return []
    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)
    
    scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    
    print("\n[RERANK] Ranking Shift Analysis:")
    for i, (score, doc) in enumerate(scored_docs[:top_n]):
        orig_idx = documents.index(doc) + 1
        page = doc.metadata.get("page", "?")
        change = "UP" if (i+1) < orig_idx else ("DOWN" if (i+1) > orig_idx else "SAME")
        print(f"  #{i+1} Page {page} | Score: {score:.3f} | Orig Path: #{orig_idx} ({change})")
        
    return [doc for score, doc in scored_docs[:top_n]]

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

def _build_retriever(collection_name: str, filter_dict: dict = None, k: int = 5):
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    
    chunks = vectorstore.get()
    documents = [
        Document(page_content=content, metadata=metadata)
        for content, metadata in zip(chunks["documents"], chunks["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(documents)
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vectorstore.as_retriever(search_kwargs={"filter": filter_dict, "k": k}), bm25_retriever],
        weights=[0.6, 0.4]
    )
    return ensemble_retriever, vectorstore

def _enrich_with_related_rules(
    docs: list[Document], vectorstore: Chroma
) -> list[Document]:
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
    print(f"\n{'='*15} RAG REQUEST (RERANK MODE) {'='*15}")
    collection = _collection_name(book_name)
    llm = get_llm(provider=provider, api_key=api_key, model=model)

    chroma_filter = None
    if mode == "generate_task":
        chroma_filter = {"content_type": "exercise"}
    elif mode == "question":
        chroma_filter = {"content_type": {"$in": ["rule", "reference", "example", "vocabulary"]}}
    
    retriever, vectorstore = _build_retriever(collection, chroma_filter, k=20)

    print(f"[RAG] Stage 1: Retrieving candidates from {book_name}...")
    
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
        docs = retriever.invoke(search_question)
        docs = _enrich_with_related_rules(docs, vectorstore)
        for doc in docs:
            page = str(doc.metadata.get("page", "?"))
            book_id = doc.metadata.get("book", "unknown")
            if book_id.lower().endswith(".pdf"): book_id = book_id[:-4]
            result = extract_and_format_task(doc.page_content, book_id, page, llm=llm)
            if result and result != "NO_TASK_FOUND":
                return result, _build_active_task_context(result, page, book_id)
        return "No exercise found.", active_task

    raw_docs = retriever.invoke(search_question) 
    if not raw_docs:
        return "No relevant information found.", active_task

    print(f"[RAG] Stage 2: Reranking {len(raw_docs)} candidates...")
    docs = _rerank_documents(search_question, raw_docs, top_n=5)
    
    docs = _enrich_with_related_rules(docs, vectorstore)
    context = _format_context(docs)

    print("[RAG] Stage 3: LLM generation...")
    if mode == "homework":
        chain = HOMEWORK_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question, "active_task_context": active_task_str})
    else:
        chain = QUESTION_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "chat_history": history_str, "question": question, "active_task_context": active_task_str})

    print(f"{'='*15} RAG REQUEST END {'='*15}\n")
    return answer, active_task