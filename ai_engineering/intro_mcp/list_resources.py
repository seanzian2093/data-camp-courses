import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def list_resources():
    """List all available resources from the MCP server."""
    params = StdioServerParameters(command=sys.executable, args=["currency_server.py"])

    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            # Get the list of resources
            response = await session.list_resources()

            print("Available resources:")
            # Print each resource's URI, name, and description
            for resource in response.resources:
                print(f" - {resource.uri}")
                print(f"   Name: {resource.name}")
                print(f"   Description: {resource.description}")

            return response.resources
asyncio.run(list_resources())