"""FastAPI application, served by uvicorn.

The web layer is deliberately thin: it reads what the worker wrote and accepts operator
overrides. No inference happens in a request handler.
"""
