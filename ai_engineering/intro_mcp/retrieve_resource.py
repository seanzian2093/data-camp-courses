import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Define an async function for reading MCP resources
async def read_resource(resource_uri: str):
    """Read a specific resource by URI."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["currency_server.py"],
    )

    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()

            print(f"Reading resource: {resource_uri}")
            # Read the resource from the session context
            resource_content = await session.read_resource(resource_uri)

            # Print the contents of each resource
            for content in resource_content.contents:
                print(f"\nContent ({content.mime_type}):")
                print(content.text)

            return resource_content

asyncio.run(read_resource("file://currencies.txt"))