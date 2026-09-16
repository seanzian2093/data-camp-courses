import sys
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def call_mcp_tool(tool_name: str, arguments: dict) -> str:
    params = StdioServerParameters(
        command=sys.executable,
        args=["currency_server.py"],
    )

    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()

            # Call the currency conversion tool
            result = await session.call_tool(tool_name, arguments)
            
            # Extract and print the text content of the server response
            text_content = result.content[0].text

            print(f"Conversion Result: {text_content}")
            return str(text_content)
            

# Run the "convert_currency" tool
asyncio.run(
    call_mcp_tool("convert_currency",
                  {"amount": 250.0, "from_currency": "USD", "to_currency": "EUR"})
)