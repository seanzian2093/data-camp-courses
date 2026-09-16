import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def list_prompts():
    """List all available prompts from the MCP server."""
    params = StdioServerParameters(command=sys.executable, args=["currency_server.py"])

    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()

            # List available prompts
            prompts = await session.list_prompts()
            print(f"Available prompts: {[p.name for p in prompts.prompts]}")

            return prompts.prompts

asyncio.run(list_prompts())