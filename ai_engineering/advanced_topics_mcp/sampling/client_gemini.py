import os
import asyncio
import sys
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.session import ClientRequestContext
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    TextContent,
    SamplingMessage,
)

genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model = "gemini-3.6-flash"

server_params = StdioServerParameters(
    command=sys.executable,
    args=["server.py"],
)


async def chat(input_messages: list[SamplingMessage], max_tokens=4000):
    contents = []
    for msg in input_messages:
        if msg.role in ("user", "assistant") and msg.content.type == "text":
            content = (
                msg.content.text if hasattr(msg.content, "text") else str(msg.content)
            )
            role = "model" if msg.role == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=content)]))

    response = genai_client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(max_output_tokens=max_tokens),
    )

    return response.text or ""


async def sampling_callback(
    context: ClientRequestContext, params: CreateMessageRequestParams
):
    # Call Gemini using the Google GenAI SDK
    text = await chat(params.messages)

    return CreateMessageResult(
        role="assistant",
        model=model,
        content=TextContent(type="text", text=text),
    )


async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write, sampling_callback=sampling_callback
        ) as session:
            await session.initialize()

            while True:
                text_to_summarize = input("\nText to summarize (or 'quit'): ")
                if text_to_summarize.strip().lower() in ("quit", "exit"):
                    break

                result = await session.call_tool(
                    name="summarize",
                    arguments={"text_to_summarize": text_to_summarize},
                )
                print(result.content)


if __name__ == "__main__":
    asyncio.run(run())
