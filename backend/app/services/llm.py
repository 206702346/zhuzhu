import requests
from app.core.config import settings


def build_prompt(question: str, contexts: list) -> str:
    context_text = ""

    for i, item in enumerate(contexts, start=1):
        source = item.get("source", "unknown")
        chunk_id = item.get("chunk_id", -1)
        text = item.get("text", "")

        context_text += f"\n[引用 {i}] 来源: {source}, 片段: {chunk_id}\n{text}\n"

    prompt = f"""
你是一个课程知识库问答助手。请严格基于给定的参考资料回答问题。

要求：
1. 如果参考资料中没有答案，请明确说“根据当前知识库资料无法回答”。
2. 不要编造参考资料中没有的信息。
3. 回答尽量清晰、结构化。
4. 回答末尾列出你参考了哪些引用编号。

参考资料：
{context_text}

用户问题：
{question}

请给出回答：
"""
    return prompt.strip()


def call_llm(prompt: str) -> str:
    """
    第一版采用 OpenAI-compatible API。
    如果你暂时没有 key，可以先返回 prompt 或 mock answer。
    """
    if not settings.llm_api_key or not settings.llm_base_url or not settings.llm_model_name:
        return (
            "当前未配置大模型 API，因此返回 mock 回答。\n\n"
            "你已经成功完成了检索部分。以下是构造出的 Prompt：\n\n"
            + prompt[:3000]
        )

    url = settings.llm_base_url.rstrip("/") + "/chat/completions"

    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": settings.llm_model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.2,
    }

    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]