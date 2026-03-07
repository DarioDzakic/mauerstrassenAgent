"""Finnhub MCP Server — exposes financial data tools over stdio transport."""

import os
import httpx
from mcp.server.fastmcp import FastMCP

FINNHUB_BASE = "https://finnhub.io/api/v1"
API_KEY = os.getenv("FINNHUB_API_KEY", "")

mcp = FastMCP("finnhub-financial-data")


def _get(endpoint: str, params: dict | None = None) -> dict:
    """Make an authenticated GET request to the Finnhub API."""
    params = params or {}
    params["token"] = API_KEY
    resp = httpx.get(f"{FINNHUB_BASE}{endpoint}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


@mcp.tool()
def get_company_profile(symbol: str) -> dict:
    """Get company profile: name, sector, industry, market cap, exchange.

    Args:
        symbol: Stock ticker symbol (e.g. 'AAPL')
    """
    return _get("/stock/profile2", {"symbol": symbol.upper()})


@mcp.tool()
def get_basic_financials(symbol: str) -> dict:
    """Get basic financial metrics: P/E, P/B, ROE, margins, 52-week range, dividend yield.

    Args:
        symbol: Stock ticker symbol (e.g. 'AAPL')
    """
    return _get("/stock/metric", {"symbol": symbol.upper(), "metric": "all"})


@mcp.tool()
def get_financials_reported(symbol: str, freq: str = "annual") -> dict:
    """Get as-reported financials: income statement, balance sheet, cash flow.

    Args:
        symbol: Stock ticker symbol (e.g. 'AAPL')
        freq: Frequency — 'annual' or 'quarterly'
    """
    return _get(
        "/stock/financials-reported",
        {"symbol": symbol.upper(), "freq": freq},
    )


@mcp.tool()
def search_in_filing(symbol: str, query: str) -> dict:
    """Search inside SEC filings (10-K/10-Q) for specific keywords.

    Args:
        symbol: Stock ticker symbol (e.g. 'AAPL')
        query: Search keyword (e.g. 'risk factors', 'revenue growth')
    """
    return _get(
        "/stock/search-in-filing",
        {"symbol": symbol.upper(), "query": query},
    )


@mcp.tool()
def get_filing_sentiment(symbol: str) -> dict:
    """Get sentiment analysis of the latest 10-K filing vs prior year.

    Args:
        symbol: Stock ticker symbol (e.g. 'AAPL')
    """
    return _get("/stock/filings-sentiment", {"symbol": symbol.upper()})


if __name__ == "__main__":
    mcp.run(transport="stdio")
