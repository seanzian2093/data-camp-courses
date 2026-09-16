import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def read_prompt(user_input: str = "How much is 50 GBP in euros?", prompt_name: str = "convert_currency_prompt") -> str:
    """Retrieve a prompt from the MCP server with user input."""
    params = StdioServerParameters(command=sys.executable, args=["currency_server.py"])

    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()

            # Retrieve the prompt with the user's input
            prompt = await session.get_prompt(prompt_name, arguments={"currency_request": user_input})

            # Print the full prompt text (template + user request)
            text = prompt.messages[0].content.text
            print(text)
            return text

asyncio.run(read_prompt(user_input="How much is 50 GBP in euros?"))