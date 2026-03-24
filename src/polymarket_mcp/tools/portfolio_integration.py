"""
Portfolio tools integration for server.py.

This module provides helper functions to integrate portfolio tools into the MCP server.
"""
import mcp.types as types
from .portfolio import PORTFOLIO_TOOLS


def _validate_rate_limiter(rate_limiter) -> None:
    """Ensure portfolio tools receive a RateLimiter-compatible dependency."""
    acquire = getattr(rate_limiter, "acquire", None)
    if not callable(acquire):
        raise TypeError(
            "Portfolio tools require a rate_limiter with an acquire() method. "
            f"Got {type(rate_limiter).__name__}."
        )


def get_portfolio_tool_definitions() -> list[types.Tool]:
    """
    Get portfolio tool definitions for MCP server.

    Returns:
        List of 8 portfolio tools as MCP Tool objects
    """
    tools = []

    for tool_def in PORTFOLIO_TOOLS:
        tools.append(types.Tool(
            name=tool_def["name"],
            description=tool_def["description"],
            inputSchema=tool_def["inputSchema"]
        ))

    return tools


async def call_portfolio_tool(name: str, arguments: dict, polymarket_client, rate_limiter, config) -> list[types.TextContent]:
    """
    Call a portfolio tool by name.

    Args:
        name: Tool name
        arguments: Tool arguments
        polymarket_client: PolymarketClient instance
        rate_limiter: RateLimiter instance
        config: PolymarketConfig instance

    Returns:
        List of TextContent with tool results

    Raises:
        ValueError: If tool name is unknown
    """
    # Find the tool handler
    tool_handler = None
    for tool_def in PORTFOLIO_TOOLS:
        if tool_def["name"] == name:
            tool_handler = tool_def["handler"]
            break

    if not tool_handler:
        raise ValueError(f"Unknown portfolio tool: {name}")

    _validate_rate_limiter(rate_limiter)

    # Call the handler with required dependencies
    result = await tool_handler(
        polymarket_client=polymarket_client,
        rate_limiter=rate_limiter,
        config=config,
        **arguments
    )

    return result
