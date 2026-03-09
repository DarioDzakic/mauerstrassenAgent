"""Finnhub MCP Server — exposes financial data tools over stdio transport."""

import os
import httpx
from mcp.server.fastmcp import FastMCP

from dotenv import load_dotenv
load_dotenv()

FINNHUB_BASE = "https://finnhub.io/api/v1"
API_KEY = os.getenv("FINNHUB_API_KEY")

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
def get_company_news(symbol: str, from_date: str, to_date: str) -> list:
    """Get latest company news articles. Free tier includes 1 year of history.

    Args:
        symbol: Stock ticker symbol (e.g. 'AAPL')
        from_date: Start date in YYYY-MM-DD format
        to_date: End date in YYYY-MM-DD format
    """
    return _get(
        "/company-news",
        {"symbol": symbol.upper(), "from": from_date, "to": to_date},
    )


@mcp.tool()
def get_recommendation_trends(symbol: str) -> list:
    """Get latest analyst recommendation trends (buy, hold, sell counts).

    Args:
        symbol: Stock ticker symbol (e.g. 'AAPL')
    """
    return _get("/stock/recommendation", {"symbol": symbol.upper()})


@mcp.tool()
def get_quote(symbol: str) -> dict:
    """Get real-time stock quote: current price, change, high/low, open, previous close.

    Args:
        symbol: Stock ticker symbol (e.g. 'AAPL')
    """
    return _get("/quote", {"symbol": symbol.upper()})


if __name__ == "__main__":
    mcp.run(transport="stdio")
