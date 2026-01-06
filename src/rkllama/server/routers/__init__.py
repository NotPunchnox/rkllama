"""FastAPI routers for RKLlama server."""

from rkllama.server.routers import modelfile, native, ollama, openai

__all__ = ["native", "ollama", "openai", "modelfile"]
