from dotenv import load_dotenv
# 加载环境变量
load_dotenv(verbose=True)
import json
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
mcp = FastMCP("search")

@mcp.tool()
async def tavily_search(query: str) -> str:
    """一个针对全面、准确和可信的结果进行了优化的搜索引擎。当需要回答有关时事的问题时很有用。输入应该是搜索查询。"""
    client = TavilyClient()
    search_r = client.search(query=query, max_results=2)
    return json.dumps(search_r, ensure_ascii=False)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")