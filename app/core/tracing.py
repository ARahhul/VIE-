import os

from app.core.config import settings


def configure_tracing() -> None:
    """Points LangGraph/LangChain's built-in tracing at LangSmith via env vars.

    This is the only tracing setup in the app — nodes must not call
    langsmith.Client().create_run(...) themselves. With these env vars set,
    every node in the compiled StateGraph reports to one nested trace per
    graph.invoke() call automatically.
    """
    if not settings.langchain_tracing_v2:
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
