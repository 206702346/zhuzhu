from typing import List


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[str]:
    """
    简单按字符长度切分文本。
    第一版先这样，后续可以升级成按标题、段落、句子切分。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == text_len:
            break

        start = end - chunk_overlap

    return chunks