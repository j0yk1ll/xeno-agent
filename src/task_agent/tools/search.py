import logging
import aiohttp
import asyncio

from src.utils.tool import Tool

class SearchTool(Tool):
    name = "search"
    description = "Returns a bullet list of search results. Each result contains the title, url and a short content preview of a website. For the full content use the browser tool."
    inputs = {
        "query": {"type": "string", "description": "The search query."}
    }
    output_type = "string"

    def forward(self, query: str) -> str:
        logging.info(f"🧰 Using tool: {self.name}")
        return asyncio.run(self._search(query))

    async def _search(self, query: str):
        async with aiohttp.ClientSession() as session:
            async with session.post("http://localhost:8080/search", data={"q": query, "format": "json"}) as response:
                json_response = await response.json()

                response_string = "\n".join([
                    f"-\n\ttitle: {result['title']}\n\turl: {result['url']}\n\tpreview: {result['content']}"
                    for result in json_response["results"]
                ])

                return response_string
