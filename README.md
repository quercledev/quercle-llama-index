# llama-index-tools-quercle

Quercle web search, fetch, and extraction tools for [LlamaIndex](https://docs.llamaindex.ai/).

## Installation

```bash
uv add llama-index-tools-quercle
# or
pip install llama-index-tools-quercle
```

## Setup

Set your API key as an environment variable:

```bash
export QUERCLE_API_KEY=qk_...
```

Get your API key at [quercle.dev](https://quercle.dev).

## Quick Start

```python
from llama_index.tools.quercle import QuercleToolSpec

spec = QuercleToolSpec()
tools = spec.to_tool_list()
# tools contains FunctionTool instances for all 5 tools:
# search, fetch, raw_search, raw_fetch, extract
```

## Tools

### `search` / `asearch` -- AI-Synthesized Web Search

Searches the web and returns an AI-synthesized answer with citations.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | `str` | Yes | Search query |
| `allowed_domains` | `list[str]` | No | Only include results from these domains |
| `blocked_domains` | `list[str]` | No | Exclude results from these domains |

### `fetch` / `afetch` -- Fetch URL and Analyze with AI

Fetches a URL and processes its content with an AI prompt.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `url` | `str` | Yes | URL to fetch |
| `prompt` | `str` | Yes | Instructions for how to process the page content |

### `raw_search` / `araw_search` -- Raw Web Search

Searches the web and returns raw search results without AI synthesis.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | `str` | Yes | Search query |
| `format` | `str` | No | Response format (`"markdown"` or `"json"`) |
| `use_safeguard` | `bool` | No | Enable content safety filtering |

### `raw_fetch` / `araw_fetch` -- Raw URL Content

Fetches a URL and returns its raw content without AI processing.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `url` | `str` | Yes | URL to fetch |
| `format` | `str` | No | Response format (`"markdown"` or `"html"`) |
| `use_safeguard` | `bool` | No | Enable content safety filtering |

### `extract` / `aextract` -- Extract Relevant Content from URL

Fetches a URL and returns only the chunks relevant to a query.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `url` | `str` | Yes | URL to fetch |
| `query` | `str` | Yes | Query describing what content to extract |
| `format` | `str` | No | Response format (`"markdown"` or `"json"`) |
| `use_safeguard` | `bool` | No | Enable content safety filtering |

## Direct Tool Usage

### Sync

```python
from llama_index.tools.quercle import QuercleToolSpec

spec = QuercleToolSpec()

# AI-synthesized search
result = spec.search(query="best practices for building AI agents")
print(result)

# Search with domain filtering
result = spec.search(
    query="Python documentation",
    allowed_domains=["docs.python.org"],
)
print(result)

# Fetch and analyze a page with AI
result = spec.fetch(
    url="https://en.wikipedia.org/wiki/Python_(programming_language)",
    prompt="Summarize the key features of Python",
)
print(result)

# Raw search results as JSON
result = spec.raw_search(query="LlamaIndex tutorials", format="json")
print(result)

# Raw page content as markdown
result = spec.raw_fetch(
    url="https://en.wikipedia.org/wiki/Python_(programming_language)",
    format="markdown",
)
print(result)

# Extract relevant content from a page
result = spec.extract(
    url="https://example.com/pricing",
    query="pricing plans and features",
    format="json",
)
print(result)
```

### Async

```python
import asyncio
from llama_index.tools.quercle import QuercleToolSpec

async def main():
    spec = QuercleToolSpec()

    result = await spec.asearch(query="latest AI agent frameworks")
    print(result)

    result = await spec.afetch(
        url="https://en.wikipedia.org/wiki/TypeScript",
        prompt="What is TypeScript?",
    )
    print(result)

    result = await spec.araw_search(query="LlamaIndex tutorials", format="json")
    print(result)

    result = await spec.araw_fetch(
        url="https://en.wikipedia.org/wiki/TypeScript",
        format="markdown",
    )
    print(result)

    result = await spec.aextract(
        url="https://example.com/pricing",
        query="pricing plans and features",
    )
    print(result)

asyncio.run(main())
```

### Standalone Tools

```python
from llama_index.tools.quercle import (
    create_quercle_search_tool,
    create_quercle_fetch_tool,
    create_quercle_raw_search_tool,
    create_quercle_raw_fetch_tool,
    create_quercle_extract_tool,
)

search_tool = create_quercle_search_tool()
fetch_tool = create_quercle_fetch_tool()
raw_search_tool = create_quercle_raw_search_tool()
raw_fetch_tool = create_quercle_raw_fetch_tool()
extract_tool = create_quercle_extract_tool()
```

### Custom API Key

```python
spec = QuercleToolSpec(api_key="qk_...")
```

## Agentic Usage

### With FunctionAgent

```python
import asyncio
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.tools.quercle import QuercleToolSpec

async def main():
    spec = QuercleToolSpec()
    tools = spec.to_tool_list()

    agent = FunctionAgent(
        tools=tools,
        llm=OpenAI(model="gpt-4o"),
        system_prompt="You are a helpful research assistant. Use the search, fetch, "
        "and extract tools to find accurate, up-to-date information.",
    )

    response = await agent.run(
        user_msg="Research the latest developments in WebAssembly and summarize them"
    )
    print(response)

asyncio.run(main())
```

### With ReActAgent

```python
from llama_index.core.agent.workflow import ReActAgent

agent = ReActAgent(
    tools=spec.to_tool_list(),
    llm=OpenAI(model="gpt-4o"),
    verbose=True,
)

response = await agent.run(user_msg="Search for trending AI papers this week")
```

### Streaming

```python
from llama_index.core.agent.workflow import AgentStream

handler = agent.run(user_msg="Summarize the latest AI news")
async for event in handler.stream_events():
    if isinstance(event, AgentStream):
        print(event.delta, end="", flush=True)
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `api_key` | `QUERCLE_API_KEY` env var | Your Quercle API key |
| `timeout` | `None` | Request timeout in seconds |

## API Reference

| Export | Description |
|---|---|
| `QuercleToolSpec` | LlamaIndex `BaseToolSpec` with `search`, `fetch`, `raw_search`, `raw_fetch`, `extract` (+ async variants) |
| `create_quercle_search_tool(...)` | Standalone `FunctionTool` -- AI-synthesized web search with citations |
| `create_quercle_fetch_tool(...)` | Standalone `FunctionTool` -- Fetch a URL and analyze content with AI |
| `create_quercle_raw_search_tool(...)` | Standalone `FunctionTool` -- Raw web search results (markdown/json) |
| `create_quercle_raw_fetch_tool(...)` | Standalone `FunctionTool` -- Raw URL content (markdown/html) |
| `create_quercle_extract_tool(...)` | Standalone `FunctionTool` -- Extract relevant content from a URL |

All tools use the `QUERCLE_API_KEY` environment variable by default. Use `api_key` parameter to provide a custom key.

## License

MIT
