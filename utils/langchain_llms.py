from langchain_ollama import ChatOllama
from langchain_core.tools import Tool


class LangchainLLama:
    def __init__(self, model: str = "llama3.2"):
        self.model = ChatOllama(model=model)

    def get_response(self, prompt: str):
        message = self.model.invoke(prompt)
        return message.content

    def get_response_with_tools(self, prompt: str, tools: list[Tool]):
        return self.model.invoke(prompt, tools=tools)
