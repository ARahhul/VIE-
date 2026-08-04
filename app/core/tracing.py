import os

from app.core.config import settings


def configure_tracing() -> None:
    """Points LangGraph/LangChain's built-in tracing at LangSmith, Langfuse's
    @observe decorators at their credentials, and Google's ADC discovery at
    the Vertex AI service account — all via env vars.

    This is the only tracing/credential-export setup in the app — nodes
    must not call langsmith.Client().create_run(...) or manually construct
    Langfuse spans themselves. With these env vars set, every node in the
    compiled StateGraph reports to one nested LangSmith trace per
    graph.invoke() call, every @observe-decorated video-LLM backend call
    reports to Langfuse, and genai.Client(vertexai=True, ...) can find
    credentials, all automatically.

    Must run before anything imports langfuse.decorators or constructs a
    Vertex-mode genai.Client — pydantic-settings reads .env into our own
    Settings object but never populates real os.environ, and both
    Langfuse's decorator-based global client and google-auth's ADC
    discovery read their respective env vars directly from os.environ, not
    from Settings. Skipping this call is why @observe silently no-ops (no
    exception, just zero traces delivered) and why Vertex auth would fail
    with a credentials-not-found error despite google_application_credentials
    being set correctly in .env.

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

    if settings.google_genai_use_vertexai:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        if settings.google_cloud_project:
            os.environ["GOOGLE_CLOUD_PROJECT"] = settings.google_cloud_project
        os.environ["GOOGLE_CLOUD_LOCATION"] = settings.google_cloud_location
        if settings.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
