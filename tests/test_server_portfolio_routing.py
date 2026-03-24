"""Regression tests for portfolio tool routing and dependency validation."""

from unittest.mock import AsyncMock, MagicMock

import mcp.types as types
import pytest

from polymarket_mcp import server as server_module
from polymarket_mcp.tools import portfolio_integration


@pytest.mark.asyncio
async def test_server_routes_portfolio_tools_with_rate_limiter(monkeypatch):
    """Portfolio tools should receive a rate limiter, not safety limits."""
    fake_client = MagicMock()
    fake_config = MagicMock()
    fake_safety_limits = MagicMock()
    fake_rate_limiter = MagicMock()
    captured = {}

    async def fake_call_portfolio_tool(name, arguments, polymarket_client, rate_limiter, config):
        captured["name"] = name
        captured["arguments"] = arguments
        captured["polymarket_client"] = polymarket_client
        captured["rate_limiter"] = rate_limiter
        captured["config"] = config
        return [types.TextContent(type="text", text="ok")]

    monkeypatch.setattr(server_module, "polymarket_client", fake_client)
    monkeypatch.setattr(server_module, "config", fake_config)
    monkeypatch.setattr(server_module, "safety_limits", fake_safety_limits)
    monkeypatch.setattr(server_module, "get_rate_limiter", lambda: fake_rate_limiter)
    monkeypatch.setattr(
        server_module.portfolio_integration,
        "call_portfolio_tool",
        fake_call_portfolio_tool,
    )

    result = await server_module.call_tool("get_portfolio_value", {"include_breakdown": True})

    assert result == [types.TextContent(type="text", text="ok")]
    assert captured["name"] == "get_portfolio_value"
    assert captured["arguments"] == {"include_breakdown": True}
    assert captured["polymarket_client"] is fake_client
    assert captured["config"] is fake_config
    assert captured["rate_limiter"] is fake_rate_limiter
    assert captured["rate_limiter"] is not fake_safety_limits


@pytest.mark.asyncio
async def test_call_portfolio_tool_rejects_invalid_rate_limiter(monkeypatch):
    """Invalid rate limiter dependencies should fail with a clear error."""
    tool_handler = AsyncMock(return_value=[types.TextContent(type="text", text="ok")])
    monkeypatch.setattr(
        portfolio_integration,
        "PORTFOLIO_TOOLS",
        [
            {
                "name": "test_portfolio_tool",
                "description": "test",
                "inputSchema": {"type": "object", "properties": {}},
                "handler": tool_handler,
            }
        ],
    )

    with pytest.raises(TypeError, match="rate_limiter"):
        await portfolio_integration.call_portfolio_tool(
            "test_portfolio_tool",
            {},
            MagicMock(),
            {"max_order_size_usd": 1000},
            MagicMock(),
        )

    tool_handler.assert_not_called()
