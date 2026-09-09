import sys
import asyncio

from fastmcp import FastMCP
# ============================================================
# FastMCP Server
# ============================================================

mcp = FastMCP("Research MCP Server")


@mcp.tool
def get_weather(city: str) -> str:
    """Return mock weather information for a city."""

    weather = {
        "kolkata": "32°C, partly cloudy",
        "delhi": "35°C, sunny",
        "mumbai": "30°C, humid",
        "bangalore": "26°C, cloudy",
    }

    return weather.get(
        city.lower().strip(),
        f"Weather information not available for {city}.",
    )


@mcp.tool
def get_news(topic: str) -> str:
    """Return mock news information about a topic."""

    news = {
        "ai": (
            "AI companies are developing new generative AI "
            "systems and investing in AI infrastructure."
        ),
        "technology": (
            "Technology companies are investing heavily in "
            "AI, cloud computing and automation."
        ),
        "science": (
            "Researchers continue to make progress in AI "
            "and scientific computing."
        ),
    }

    return news.get(
        topic.lower().strip(),
        f"No news available for {topic}.",
    )


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8000
    )

