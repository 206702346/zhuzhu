#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List


MODE_ORDER = {
    "current": 0,
    "hybrid": 1,
    "vector": 2,
    "bm25": 3,
}


def load_reports(report_paths: List[Path]) -> List[Dict[str, Any]]:
    """
    从多个 report.json 中提取统一记录：
    - dataset
    - top_k
    - mode
    - summary metrics
    """
    records = []

    for report_path in report_paths:
        if not report_path.exists():
            raise FileNotFoundError(f"报告文件不存在: {report_path}")

        report = json.loads(report_path.read_text(encoding="utf-8"))
        meta = report.get("meta", {})
        data_path = meta.get("data", "")
        dataset = Path(data_path).stem if data_path else report_path.stem
        top_k = int(meta.get("top_k", 0))

        results = report.get("results", {})
        for mode, payload in results.items():
            s = payload.get("summary", {})
            records.append({
                "dataset": dataset,
                "top_k": top_k,
                "mode": mode,
                "samples": s.get("samples", 0),
                "doc_hit@1": s.get("doc_hit@1", 0.0),
                "doc_hit@k": s.get("doc_hit@k", 0.0),
                "strict_hit@1": s.get("strict_hit@1", 0.0),
                "strict_hit@k": s.get("strict_hit@k", 0.0),
                "doc_mrr@k": s.get("doc_mrr@k", 0.0),
                "strict_mrr@k": s.get("strict_mrr@k", 0.0),
            })

    return records


def fmt(x) -> str:
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def render_table(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "mode",
        "samples",
        "doc_hit@1",
        "doc_hit@k",
        "strict_hit@1",
        "strict_hit@k",
        "doc_mrr@k",
        "strict_mrr@k",
    ]

    # 排序：current / hybrid / vector / bm25
    rows = sorted(rows, key=lambda r: (MODE_ORDER.get(r["mode"], 999), r["mode"]))

    table = []
    table.append("| " + " | ".join(headers) + " |")
    table.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for r in rows:
        table.append("| " + " | ".join([
            fmt(r["mode"]),
            fmt(r["samples"]),
            fmt(r["doc_hit@1"]),
            fmt(r["doc_hit@k"]),
            fmt(r["strict_hit@1"]),
            fmt(r["strict_hit@k"]),
            fmt(r["doc_mrr@k"]),
            fmt(r["strict_mrr@k"]),
        ]) + " |")

    return "\n".join(table)


def build_markdown(records: List[Dict[str, Any]]) -> str:
    by_dataset = defaultdict(list)
    for r in records:
        by_dataset[r["dataset"]].append(r)

    lines = []
    lines.append("## 实验对比")
    lines.append("")
    lines.append("> 说明：以下结果由脚本自动生成，建议保留该区域为自动更新区。")
    lines.append("")

    for dataset in sorted(by_dataset.keys()):
        lines.append(f"### 数据集：`{dataset}`")
        lines.append("")

        by_topk = defaultdict(list)
        for r in by_dataset[dataset]:
            by_topk[r["top_k"]].append(r)

        for top_k in sorted(by_topk.keys()):
            lines.append(f"#### `top_k = {top_k}`")
            lines.append("")
            lines.append(render_table(by_topk[top_k]))
            lines.append("")

    return "\n".join(lines)


def update_readme(readme_path: Path, new_block: str) -> None:
    start_marker = "<!-- RAG_EVAL_TABLE_START -->"
    end_marker = "<!-- RAG_EVAL_TABLE_END -->"

    if not readme_path.exists():
        raise FileNotFoundError(f"README 不存在: {readme_path}")

    content = readme_path.read_text(encoding="utf-8")

    if start_marker in content and end_marker in content:
        pattern = re.compile(
            re.escape(start_marker) + r".*?" + re.escape(end_marker),
            re.S,
        )
        replacement = f"{start_marker}\n\n{new_block}\n\n{end_marker}"
        content = pattern.sub(replacement, content)
    else:
        # 如果没找到标记，就直接追加到文件末尾
        content += "\n\n" + new_block + "\n"

    readme_path.write_text(content, encoding="utf-8")
    print(f"[INFO] README 已更新: {readme_path}")


def main():
    parser = argparse.ArgumentParser(description="将评测报告自动整理进 README")
    parser.add_argument(
        "--readme",
        type=str,
        default="README.md",
        help="README 文件路径",
    )
    parser.add_argument(
        "--reports",
        type=str,
        nargs="+",
        required=True,
        help="一个或多个 report.json 路径",
    )

    args = parser.parse_args()

    readme_path = Path(args.readme)
    report_paths = [Path(p) for p in args.reports]

    records = load_reports(report_paths)
    md_block = build_markdown(records)
    update_readme(readme_path, md_block)


if __name__ == "__main__":
    main()