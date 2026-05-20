"""
Streaming LLM using DeepSeek API (OpenAI-compatible).
"""

import json
import httpx


class StreamingLLM:
    """DeepSeek / OpenAI-compatible LLM with stream=True support."""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com", model: str = "deepseek-chat"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system_prompt = "你是船舶建造领域的专家，请用中文准确回答用户关于造船的问题。回答简洁、专业。"

    async def generate_stream(self, prompt: str):
        """
        Yield (type, text) where type is "token" or "sentence".
        type="sentence" means a complete sentence ready for TTS.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "max_tokens": 512,
            "temperature": 0.7,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", f"{self.base_url}/v1/chat/completions",
                                     headers=headers, json=payload) as resp:
                buffer = ""
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            buffer += token
                            yield ("token", token)
                    except (json.JSONDecodeError, KeyError):
                        continue

                if buffer.strip():
                    yield ("final", buffer.strip())

    async def generate_once(self, prompt: str) -> str:
        """Non-streaming: return full response text."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "max_tokens": 512,
            "temperature": 0.7,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{self.base_url}/v1/chat/completions",
                                     headers=headers, json=payload)
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
