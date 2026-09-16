import sys
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from anthropic import AsyncAnthropic

async def get_context_from_mcp(user_query: str) -> tuple[str, str]:
    params = StdioServerParameters(command=sys.executable, args=["currency_server.py"])
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            resource_result = await session.read_resource("file://currencies.txt")
            resource_text = resource_result.contents[0].text
            prompt_result = await session.get_prompt("convert_currency_prompt",
                arguments={"currency_request": user_query})
            prompt_text = prompt_result.messages[0].content.text
            return resource_text, prompt_text

async def get_tools_from_mcp():
    params = StdioServerParameters(command=sys.executable, args=["currency_server.py"])
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            response = await session.list_tools()
            return response.tools

async def call_mcp_tool(tool_name: str, arguments: dict) -> str:
    params = StdioServerParameters(command=sys.executable, args=["currency_server.py"])
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return str(result.content[0].text)

async def call_llm_with_context(user_query: str):
    """Call the LLM with resource and prompt context from MCP."""
    resource_text, prompt_text = await get_context_from_mcp(user_query)

    # Combine the resource and prompt text
    full_prompt = prompt_text + "\n\nSupported currencies:\n" + resource_text

    client = AsyncAnthropic(api_key="<ANTHROPIC_API_TOKEN>")
    mcp_tools = await get_tools_from_mcp()
    anthropic_tools = [{"name": t.name, "description": t.description or "", "input_schema": t.inputSchema} for t in mcp_tools]
    
    # Send full_prompt (as a user message) and the tools list to the model
    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": full_prompt}],
        tools=anthropic_tools,
    )

    # Return the text response
    if response.stop_reason == "end_turn":
        text = next((block.text for block in response.content if block.type == "text"), "")
        print(f"\nAssistant: {text}")
        return str(text)

    # Call the tool requested in the LLM's tool use
    if response.stop_reason == "tool_use":
        tool_use = next(block for block in response.content if block.type == "tool_use")
        result = await call_mcp_tool(tool_use.name, tool_use.input)
        followup = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": full_prompt},
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": result}]},
            ],
            tools=anthropic_tools,
        )
        final_text = next((block.text for block in followup.content if block.type == "text"), None)
        if final_text:
            print(f"\nAssistant: {final_text}")
            return str(final_text)

print("=== Ambiguous request (prompt asks for clarification) ===")
asyncio.run(call_llm_with_context("Convert some euros to dollars"))
print("\n=== Unambiguous request (model calls tool) ===")
asyncio.run(call_llm_with_context("How much is 50 GBP in euros?"))