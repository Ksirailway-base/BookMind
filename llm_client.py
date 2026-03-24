from langchain_core.embeddings import Embeddings

LOCAL_BASE_URL = "http://localhost:8080/v1"

class LocalEmbeddings(Embeddings):

    def __init__(self):
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        self._ef = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._ef(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._ef([text])[0]


_embeddings = None
_reranker = None

def get_embeddings() -> LocalEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = LocalEmbeddings()
    return _embeddings

def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    return _reranker

def get_llm(provider: str = "local", api_key: str = "", model: str = ""):
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=api_key,
            model=model or "gpt-4o-mini",
            temperature=0,
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model or "gemini-2.5-flash-lite",
            temperature=0,
        )

    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url=LOCAL_BASE_URL,
            api_key="not-needed",
            model=model or "local",
            temperature=0,
        )
