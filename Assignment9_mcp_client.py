
import asyncio
from fastmcp import Client
# ============================================================
# MCP Client
# ============================================================
MCP_URL = "http://127.0.0.1:8000/mcp"
async def run_mcp_client():

    print("\n" + "=" * 70)
    print("MCP CLIENT")
    print("=" * 70)

    async with Client(MCP_URL) as client:

        tools = await client.list_tools()

        print("\nAvailable MCP tools:")

        for tool_info in tools:
            print(f"- {tool_info.name}")

        weather = await client.call_tool(
            "get_weather",
            {
                "city": "Kolkata",
            },
        )

        print("\nWEATHER:")
        print(weather)

        news = await client.call_tool(
            "get_news",
            {
                "topic": "AI",
            },
        )

        print("\nNEWS:")
        print(news)

if __name__ == "__main__":
    asyncio.run(run_mcp_client())