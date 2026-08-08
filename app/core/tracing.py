import os

from app.core.config import settings


def configure_tracing() -> None:
    """Points LangGraph/LangChain's built-in tracing at LangSmith, and
    Langfuse's @observe decorators at their credentials — both via env vars.

    This is the only tracing/credential-export setup in the app — nodes
    must not call langsmith.Client().create_run(...) or manually construct
    Langfuse spans themselves. With these env vars set, every node in the
    compiled StateGraph reports to one nested LangSmith trace per
    graph.invoke() call, and every @observe-decorated video-LLM backend call
    reports to Langfuse, automatically.

    Must run before anything imports langfuse.decorators — pydantic-settings
    reads .env into our own Settings object but never populates real
    os.environ, and Langfuse's decorator-based global client reads
    LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST directly from os.environ, not from
    Settings. Skipping this call is why @observe silently no-ops: no
    exception, just zero traces delivered.

    Any process that imports app.llm.backends outside the FastAPI lifespan
    (scripts, benchmarks) must call this explicitly first.
    """
    if settings.langchain_tracing_v2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
        if settings.langchain_api_key:
            os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key

    if settings.langfuse_public_key and settings.langfuse_secret_key:
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
        os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
        os.environ["LANGFUSE_HOST"] = settings.langfuse_host
