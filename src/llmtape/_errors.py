import os


class LLMTapeError(Exception):
    """Base class for all LLMTape errors."""


class CassetteNotFoundError(LLMTapeError):
    def __init__(self, function_name: str, fingerprint: str, cassette_path: str, request: dict):
        # Only show request content if explicitly opted in — CI logs may be public
        if os.environ.get("LLMTAPE_VERBOSE_ERRORS"):
            import json
            req_detail = f"\n\nNormalized request keys: {list(request.keys())}\n" \
                         f"(set LLMTAPE_VERBOSE_ERRORS=1 to see full request)"
        else:
            req_detail = f"\n\nRequest keys: {list(request.keys())}\n" \
                         f"(set LLMTAPE_VERBOSE_ERRORS=1 to see full request in error)"
        super().__init__(
            f"\n\nNo cassette found for '{function_name}' (fingerprint: {fingerprint[:12]})\n"
            f"Expected: {cassette_path}"
            f"{req_detail}\n"
            f"To record: LLMTAPE_MODE=record-missing pytest\n"
            f"To record all: LLMTAPE_MODE=record pytest"
        )


class TapeStreamingError(LLMTapeError, NotImplementedError):
    def __init__(self):
        super().__init__(
            "LLMTape does not support streaming calls (stream=True) in v0. "
            "Remove stream=True from the call inside your @llmtape.tape function, "
            "or use vcrpy for HTTP-level recording of streaming responses."
        )


class TapeUnsupportedProviderError(LLMTapeError):
    def __init__(self, response_type: str):
        super().__init__(
            f"LLMTape does not know how to handle response type '{response_type}'. "
            f"Supported: openai.types.chat.ChatCompletion, anthropic.types.Message. "
            f"Open an issue at https://github.com/zer0contextlost/llmtape"
        )
