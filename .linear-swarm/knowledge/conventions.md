# Coding Conventions — joplin-mcp

## File naming
- Python: snake_case for modules and functions, PascalCase for classes
- Tests mirror source: `src/joplin_mcp/tools/notes.py` → `tests/test_tools_notes.py`

## Package structure
- MCP tools live in `src/joplin_mcp/tools/` — one file per domain (notes, notebooks, tags, brain_dump)
- Vector/embedding code in `src/joplin_mcp/vector/`
- Server wiring in `src/joplin_mcp/server.py` and `fastmcp_server.py`

## HTTP client
- Use `httpx` (async) for all HTTP calls
- Joplin Data API base URL comes from config: `JOPLIN_DATA_API_URL` (default: `http://joplin-data-api:41185`)
- Do NOT use `joppy` — use direct HTTP to joplin-data-api with nginx auth injection

## Tool registration
- Tools are registered via `@mcp.tool()` decorator (FastMCP)
- Each tool function should have a clear docstring (used as tool description for AI clients)
- Pydantic models in `src/joplin_mcp/types/` for input/output schemas

## Testing
- pytest with pytest-asyncio
- Async tests use `@pytest.mark.asyncio`
- Fixtures in `tests/conftest.py` and `src/joplin_mcp/conftest.py`
- Mock HTTP calls with `pytest-mock` — do not make real API calls in unit tests
- Coverage threshold: 20% minimum (configured in pyproject.toml)

## Imports
- Standard library first, third-party second, local last
- Absolute imports preferred
- Type annotations used throughout

## Error handling
- Raise descriptive exceptions for API errors
- Log errors but don't swallow them silently

## Linting / formatting
- `black` for formatting (line-length 88)
- `ruff` for linting (E, W, F, I, B, C4, UP rules)
- `mypy` for type checking
