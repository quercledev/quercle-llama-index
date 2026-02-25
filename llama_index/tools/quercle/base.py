"""Quercle tool implementations for LlamaIndex."""

import json
from typing import Annotated

from llama_index.core.tools import FunctionTool
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from pydantic import Field

from quercle import (
    AsyncQuercleClient,
    QuercleClient,
    tool_metadata,
)


class QuercleToolSpec(BaseToolSpec):
    """Quercle tool specification for web fetch and search.

    This tool spec provides access to Quercle's AI-powered web fetch and search APIs.
    Use `to_tool_list()` to get LlamaIndex tools for use with agents.

    Example:
        >>> from llama_index.tools.quercle import QuercleToolSpec
        >>> from llama_index.core.agent import ReActAgent
        >>> from llama_index.llms.openai import OpenAI
        >>>
        >>> quercle = QuercleToolSpec(api_key="qk_...")
        >>> agent = ReActAgent.from_tools(quercle.to_tool_list(), llm=OpenAI())
        >>> response = agent.chat("Search for Python news")
    """

    spec_functions = [
        ("fetch", "afetch"),
        ("search", "asearch"),
        ("raw_fetch", "araw_fetch"),
        ("raw_search", "araw_search"),
        ("extract", "aextract"),
    ]

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize the Quercle tool spec.

        Args:
            api_key: Quercle API key. Falls back to QUERCLE_API_KEY env var if not provided.
            timeout: Request timeout in seconds.
        """
        self._api_key = api_key
        self._timeout = timeout
        self._sync_client: QuercleClient | None = None
        self._async_client: AsyncQuercleClient | None = None

    def _get_sync_client(self) -> QuercleClient:
        """Get or create the synchronous Quercle client."""
        if self._sync_client is None:
            self._sync_client = QuercleClient(
                api_key=self._api_key,
            )
        return self._sync_client

    def _get_async_client(self) -> AsyncQuercleClient:
        """Get or create the asynchronous Quercle client."""
        if self._async_client is None:
            self._async_client = AsyncQuercleClient(
                api_key=self._api_key,
            )
        return self._async_client

    def fetch(
        self,
        url: Annotated[str, Field(description=tool_metadata["fetch"]["parameters"]["url"])],
        prompt: Annotated[str, Field(description=tool_metadata["fetch"]["parameters"]["prompt"])],
    ) -> str:
        """Fetch a web page and analyze its content using AI."""
        return self._get_sync_client().fetch(url=url, prompt=prompt, timeout=self._timeout).result

    async def afetch(
        self,
        url: Annotated[str, Field(description=tool_metadata["fetch"]["parameters"]["url"])],
        prompt: Annotated[str, Field(description=tool_metadata["fetch"]["parameters"]["prompt"])],
    ) -> str:
        """Async fetch a web page and analyze its content using AI."""
        return (await self._get_async_client().fetch(
            url=url, prompt=prompt, timeout=self._timeout
        )).result

    def search(
        self,
        query: Annotated[str, Field(description=tool_metadata["search"]["parameters"]["query"])],
        allowed_domains: Annotated[
            list[str] | None,
            Field(description=tool_metadata["search"]["parameters"]["allowed_domains"]),
        ] = None,
        blocked_domains: Annotated[
            list[str] | None,
            Field(description=tool_metadata["search"]["parameters"]["blocked_domains"]),
        ] = None,
    ) -> str:
        """Search the web and get AI-synthesized answers with citations."""
        return self._get_sync_client().search(
            query,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
            timeout=self._timeout,
        ).result

    async def asearch(
        self,
        query: Annotated[str, Field(description=tool_metadata["search"]["parameters"]["query"])],
        allowed_domains: Annotated[
            list[str] | None,
            Field(description=tool_metadata["search"]["parameters"]["allowed_domains"]),
        ] = None,
        blocked_domains: Annotated[
            list[str] | None,
            Field(description=tool_metadata["search"]["parameters"]["blocked_domains"]),
        ] = None,
    ) -> str:
        """Async search the web and get AI-synthesized answers with citations."""
        return (await self._get_async_client().search(
            query,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
            timeout=self._timeout,
        )).result

    def raw_fetch(
        self,
        url: Annotated[str, Field(description=tool_metadata["raw_fetch"]["parameters"]["url"])],
        format: Annotated[
            str | None,
            Field(description=tool_metadata["raw_fetch"]["parameters"]["format"]),
        ] = None,
        use_safeguard: Annotated[
            bool | None,
            Field(description=tool_metadata["raw_fetch"]["parameters"]["use_safeguard"]),
        ] = None,
    ) -> str:
        """Fetch a URL and return raw markdown or HTML."""
        response = self._get_sync_client().raw_fetch(
            url, format=format, use_safeguard=use_safeguard, timeout=self._timeout
        )
        result = response.result
        return result if isinstance(result, str) else json.dumps(result)

    async def araw_fetch(
        self,
        url: Annotated[str, Field(description=tool_metadata["raw_fetch"]["parameters"]["url"])],
        format: Annotated[
            str | None,
            Field(description=tool_metadata["raw_fetch"]["parameters"]["format"]),
        ] = None,
        use_safeguard: Annotated[
            bool | None,
            Field(description=tool_metadata["raw_fetch"]["parameters"]["use_safeguard"]),
        ] = None,
    ) -> str:
        """Async fetch a URL and return raw markdown or HTML."""
        response = await self._get_async_client().raw_fetch(
            url, format=format, use_safeguard=use_safeguard, timeout=self._timeout
        )
        result = response.result
        return result if isinstance(result, str) else json.dumps(result)

    def raw_search(
        self,
        query: Annotated[
            str,
            Field(description=tool_metadata["raw_search"]["parameters"]["query"]),
        ],
        format: Annotated[
            str | None,
            Field(description=tool_metadata["raw_search"]["parameters"]["format"]),
        ] = None,
        use_safeguard: Annotated[
            bool | None,
            Field(description=tool_metadata["raw_search"]["parameters"]["use_safeguard"]),
        ] = None,
    ) -> str:
        """Run web search and return raw results."""
        response = self._get_sync_client().raw_search(
            query, format=format, use_safeguard=use_safeguard, timeout=self._timeout
        )
        result = response.result
        return result if isinstance(result, str) else json.dumps(result)

    async def araw_search(
        self,
        query: Annotated[
            str,
            Field(description=tool_metadata["raw_search"]["parameters"]["query"]),
        ],
        format: Annotated[
            str | None,
            Field(description=tool_metadata["raw_search"]["parameters"]["format"]),
        ] = None,
        use_safeguard: Annotated[
            bool | None,
            Field(description=tool_metadata["raw_search"]["parameters"]["use_safeguard"]),
        ] = None,
    ) -> str:
        """Async run web search and return raw results."""
        response = await self._get_async_client().raw_search(
            query, format=format, use_safeguard=use_safeguard, timeout=self._timeout
        )
        result = response.result
        return result if isinstance(result, str) else json.dumps(result)

    def extract(
        self,
        url: Annotated[str, Field(description=tool_metadata["extract"]["parameters"]["url"])],
        query: Annotated[str, Field(description=tool_metadata["extract"]["parameters"]["query"])],
        format: Annotated[
            str | None,
            Field(description=tool_metadata["extract"]["parameters"]["format"]),
        ] = None,
        use_safeguard: Annotated[
            bool | None,
            Field(description=tool_metadata["extract"]["parameters"]["use_safeguard"]),
        ] = None,
    ) -> str:
        """Fetch a URL and return chunks relevant to a query."""
        response = self._get_sync_client().extract(
            url, query, format=format, use_safeguard=use_safeguard, timeout=self._timeout
        )
        result = response.result
        return result if isinstance(result, str) else json.dumps(result)

    async def aextract(
        self,
        url: Annotated[str, Field(description=tool_metadata["extract"]["parameters"]["url"])],
        query: Annotated[str, Field(description=tool_metadata["extract"]["parameters"]["query"])],
        format: Annotated[
            str | None,
            Field(description=tool_metadata["extract"]["parameters"]["format"]),
        ] = None,
        use_safeguard: Annotated[
            bool | None,
            Field(description=tool_metadata["extract"]["parameters"]["use_safeguard"]),
        ] = None,
    ) -> str:
        """Async fetch a URL and return chunks relevant to a query."""
        response = await self._get_async_client().extract(
            url, query, format=format, use_safeguard=use_safeguard, timeout=self._timeout
        )
        result = response.result
        return result if isinstance(result, str) else json.dumps(result)


def create_quercle_fetch_tool(
    api_key: str | None = None,
    timeout: float | None = None,
) -> FunctionTool:
    """Create a standalone Quercle fetch tool.

    Args:
        api_key: Quercle API key. Falls back to QUERCLE_API_KEY env var if not provided.
        timeout: Request timeout in seconds.

    Returns:
        A FunctionTool for fetching and analyzing web pages.
    """
    spec = QuercleToolSpec(api_key=api_key, timeout=timeout)
    return FunctionTool.from_defaults(
        fn=spec.fetch,
        async_fn=spec.afetch,
        name="quercle_fetch",
        description=tool_metadata["fetch"]["description"],
    )


def create_quercle_search_tool(
    api_key: str | None = None,
    timeout: float | None = None,
) -> FunctionTool:
    """Create a standalone Quercle search tool.

    Args:
        api_key: Quercle API key. Falls back to QUERCLE_API_KEY env var if not provided.
        timeout: Request timeout in seconds.

    Returns:
        A FunctionTool for web search with AI-synthesized answers.
    """
    spec = QuercleToolSpec(api_key=api_key, timeout=timeout)
    return FunctionTool.from_defaults(
        fn=spec.search,
        async_fn=spec.asearch,
        name="quercle_search",
        description=tool_metadata["search"]["description"],
    )


def create_quercle_raw_fetch_tool(
    api_key: str | None = None,
    timeout: float | None = None,
) -> FunctionTool:
    """Create a standalone Quercle raw fetch tool.

    Args:
        api_key: Quercle API key. Falls back to QUERCLE_API_KEY env var if not provided.
        timeout: Request timeout in seconds.

    Returns:
        A FunctionTool for fetching raw markdown or HTML from URLs.
    """
    spec = QuercleToolSpec(api_key=api_key, timeout=timeout)
    return FunctionTool.from_defaults(
        fn=spec.raw_fetch,
        async_fn=spec.araw_fetch,
        name="quercle_raw_fetch",
        description=tool_metadata["raw_fetch"]["description"],
    )


def create_quercle_raw_search_tool(
    api_key: str | None = None,
    timeout: float | None = None,
) -> FunctionTool:
    """Create a standalone Quercle raw search tool.

    Args:
        api_key: Quercle API key. Falls back to QUERCLE_API_KEY env var if not provided.
        timeout: Request timeout in seconds.

    Returns:
        A FunctionTool for raw web search results.
    """
    spec = QuercleToolSpec(api_key=api_key, timeout=timeout)
    return FunctionTool.from_defaults(
        fn=spec.raw_search,
        async_fn=spec.araw_search,
        name="quercle_raw_search",
        description=tool_metadata["raw_search"]["description"],
    )


def create_quercle_extract_tool(
    api_key: str | None = None,
    timeout: float | None = None,
) -> FunctionTool:
    """Create a standalone Quercle extract tool.

    Args:
        api_key: Quercle API key. Falls back to QUERCLE_API_KEY env var if not provided.
        timeout: Request timeout in seconds.

    Returns:
        A FunctionTool for extracting relevant content chunks from URLs.
    """
    spec = QuercleToolSpec(api_key=api_key, timeout=timeout)
    return FunctionTool.from_defaults(
        fn=spec.extract,
        async_fn=spec.aextract,
        name="quercle_extract",
        description=tool_metadata["extract"]["description"],
    )
