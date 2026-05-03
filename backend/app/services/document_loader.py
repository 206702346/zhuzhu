import os
import fitz  # PyMuPDF


def load_txt_or_md(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    texts = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            texts.append(f"\n[Page {page_num + 1}]\n{text}")

    return "\n".join(texts)


def load_document(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".txt", ".md"]:
        return load_txt_or_md(file_path)

    if ext == ".pdf":
        return load_pdf(file_path)

    raise ValueError(f"暂不支持的文件类型: {ext}")