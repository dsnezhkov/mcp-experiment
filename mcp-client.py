import asyncio
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

async def get_available():
    async with sse_client("http://localhost:8000/sse") as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()

             # List available prompts
            prompts = await session.list_prompts()
            print(f"Prompts: {prompts}")

            # Get a prompt
            prompt = await session.get_prompt(
                "review_code", arguments={"code": 'print(5)'}
            )
            print(f"Prompt: {prompt}")

            # List available resources
            resources = await session.list_resources()
            print(f"Resources: {resources}")

            # List available tools
            tools = await session.list_tools()
            print(f"Tools: {tools}")

            # Read a resource
            content, mime_type = await session.read_resource("dir://desktop")
            print(f"Content: {content}")

            # Call a tool
            result = await session.call_tool("langgraph-query-tool", arguments={"query": "What is langgraph"})
            print(f"tool call result: {result}")

async def main():
    await get_available()

if __name__ == "__main__":
    asyncio.run(main())
