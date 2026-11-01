import asyncio
from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from typing import Tuple, Literal, Optional, Awaitable, Any, Callable, List, Dict

type ServerType = Literal["stdio", "sse"]


class MCPClient:
    server_params: StdioServerParameters | str
    on_connect: Callable[[ClientSession], Awaitable[None]]

    def __init__(self, server_init: Tuple[ServerType, StdioServerParameters | str]) -> None:
        self.server_params_type = server_init[0]
        if self.server_params_type == "stdio":
            self.server_params: StdioServerParameters = server_init[1]
        elif self.server_params_type == "sse":
            self.server_params = server_init[1]
        else:
            raise ValueError(f"Invalid server type: {self.server_params_type}")

    def on_connection_established(self, cb: Callable[[ClientSession], Awaitable[None]]) -> None:
        self.on_connect = cb

    async def connect_to_mcp_server(self) -> None:
        if not self.on_connect or self.server_params is None or self.on_connect is None:
            raise ValueError(
                "No callback function provided, did you set up correctly?")

        if self.server_params_type == "stdio":
            await self.connect_to_mcp_server_stdio(self.server_params, self.on_connect)
        elif self.server_params_type == "sse":
            await self.connect_to_mcp_server_sse(self.server_params, self.on_connect)
        else:
            raise ValueError(f"Invalid server type: {self.server_params_type}")
        print("ending connect to MCP server")

    @staticmethod
    def get_stdio_server_params(command: str, args: List[str], env: Dict[str, str] = None) -> StdioServerParameters:
        return StdioServerParameters(
            command=command,
            args=args,
            env=env
        )

    @staticmethod
    async def connect_to_mcp_server_stdio(
        server_params: StdioServerParameters,
        cb: Callable[[ClientSession], Awaitable[None]]
    ) -> None:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                await cb(session)

    @staticmethod
    async def connect_to_mcp_server_sse(
        server_params: str,
        cb: Callable[[ClientSession], Awaitable[None]]
    ) -> None:
        async with sse_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                await cb(session)

    @staticmethod
    async def get_tools(session: ClientSession) -> List[Tool]:
        return (await session.list_tools()).tools


if __name__ == "__main__":
    client = MCPClient(server_init=("stdio", StdioServerParameters(
        command="python",  # The command to run your server
        args=["mcp_practice.py"],  # Arguments to the command
    )))

    async def mcp_server_callback(session: ClientSession) -> None:
        print("Connected to MCP server")
        tools = await client.get_tools(session)
        print(tools)

    client.on_connection_established(mcp_server_callback)

    async def main() -> None:
        await client.connect_to_mcp_server()

    asyncio.run(main())
