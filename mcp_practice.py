from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="my-mcp-server", version="1.0.0", port=5000)


@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b


@mcp.tool()
def subtract(a: int, b: int) -> int:
    return a - b


if __name__ == "__main__":
    transport = "stdio"
    mcp.run(transport=transport)
