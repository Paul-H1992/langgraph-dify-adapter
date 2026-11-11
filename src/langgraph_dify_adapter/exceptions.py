"""Exceptions raised by the Dify adapter."""


class DifyAPIError(RuntimeError):
    """A request to the Dify chat-messages API failed or returned invalid data."""
