from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
import sys


def get_mcp_tools():

    server_params = [
        # Finnhub MCP Server — financial data via Finnhub API
        StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "mauerstrassenagent.mcp.finnhub_server",
            ]
        ),
    ]

    adapter = MCPServerAdapter(server_params)  # type: ignore

    return adapter.tools
