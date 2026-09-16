import os
import sys
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai
from google.genai import types


async def get_context_from_mcp(user_query: str) -> tuple[str, str]:
    params = StdioServerParameters(command=sys.executable, args=["currency_server.py"])
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            resource_result = await session.read_resource("file://currencies.txt")
            resource_text = resource_result.contents[0].text
            prompt_result = await session.get_prompt(
                "convert_currency_prompt", arguments={"currency_request": user_query}
            )
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

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    mcp_tools = await get_tools_from_mcp()
    gemini_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=t.name, description=t.description or "", parameters=t.input_schema
            )
            for t in mcp_tools
        ]
    )

    # Send full_prompt (as a user message) and the tools list to the model
    contents = [types.Content(role="user", parts=[types.Part(text=full_prompt)])]
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=[gemini_tool]),
    )

    candidate_content = response.candidates[0].content
    function_call = next(
        (part.function_call for part in candidate_content.parts if part.function_call),
        None,
    )

    # Return the text response
    if not function_call:
        text = response.text or ""
        print(f"\nAssistant: {text}")
        return str(text)

    # Call the tool requested in the LLM's tool use
    result = await call_mcp_tool(function_call.name, dict(function_call.args))
    contents.append(candidate_content)
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(
                    name=function_call.name, response={"result": result}
                )
            ],
        )
    )
    followup = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=[gemini_tool]),
    )
    final_text = followup.text
    if final_text:
        print(f"\nAssistant: {final_text}")
        return str(final_text)


print("=== Ambiguous request (prompt asks for clarification) ===")
asyncio.run(call_llm_with_context("Convert some euros to dollars"))
print("\n=== Unambiguous request (model calls tool) ===")
asyncio.run(call_llm_with_context("How much is 50 GBP in euros?"))
