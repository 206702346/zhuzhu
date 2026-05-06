#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
from datetime import datetime

# 让脚本能直接 import backend/app 下的模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.retriever import retrieve, hybrid_search, vector_search
from app.services.bm25_retriever import bm25_search


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return str(text).strip().lower()


def load_eval_data(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"评测文件不存在: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("评测文件必须是 JSON 数组")

    normalized = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"第 {idx + 1} 条不是对象")

        question = item.get("question")
        if not question:
            raise ValueError(f"第 {idx + 1} 条缺少 question")

        targets = item.get("targets", item.get("target", []))
        if isinstance(targets, dict):
            targets = [targets]
        if isinstance(targets, str):
            targets = [{"source": targets}]
        if not isinstance(targets, list):
            raise ValueError(f"第 {idx + 1} 条 targets 必须是数组")

        norm_targets = []
        for t in targets:
            if isinstance(t, str):
                norm_targets.append({"source": t, "chunk_id": None, "doc_id": None})
                continue

            if not isinstance(t, dict):
                raise ValueError(f"第 {idx + 1} 条 targets 里存在非法项")

            norm_targets.append({
                "source": t.get("source"),
                "chunk_id": t.get("chunk_id"),
                "doc_id": t.get("doc_id"),
            })

        normalized.append({
            "id": item.get("id", f"q{idx + 1}"),
            "question": question,
            "targets": norm_targets,
        })

    return normalized


def dispatch_retrieval(mode: str, query: str, top_k: int) -> List[Dict[str, Any]]:
    """
    mode:
      - current: 当前生产链路（hybrid + rerank + lexical）
      - hybrid : 只看 BM25 + 向量融合
      - vector : 只看向量检索
      - bm25   : 只看 BM25
    """
    mode = mode.lower().strip()

    if mode == "current":
        return retrieve(query, top_k=top_k)

    if mode == "hybrid":
        # hybrid_search 返回的是融合后的候选列表，取前 top_k 即可
        return hybrid_search(query, candidate_k=max(top_k, 10))[:top_k]

    if mode == "vector":
        return vector_search(query, top_k=max(top_k, 10))[:top_k]

    if mode == "bm25":
        return bm25_search(query, top_k=max(top_k, 10))[:top_k]

    raise ValueError(f"不支持的 mode: {mode}")


def _compact(s: str) -> str:
    return normalize_text(s).replace(" ", "")


def candidate_matches_target(
    candidate: Dict[str, Any],
    target: Dict[str, Any],
    strict: bool = False,
) -> bool:
    """
    doc-level: 只比 source / doc_id
    strict   : 除了 source / doc_id，还要求 chunk 内容命中 anchors；
               如果 target 提供 chunk_id，则也可以一起校验。
    """
    cand_source = normalize_text(candidate.get("source"))
    cand_doc_id = normalize_text(candidate.get("doc_id"))
    cand_chunk_id = candidate.get("chunk_id")
    cand_text = candidate.get("text", "")

    tgt_source = normalize_text(target.get("source"))
    tgt_doc_id = normalize_text(target.get("doc_id"))
    tgt_chunk_id = target.get("chunk_id")
    anchors = target.get("anchors") or []

    if tgt_source and cand_source != tgt_source:
        return False

    if tgt_doc_id and cand_doc_id != tgt_doc_id:
        return False

    if strict:
        if tgt_chunk_id is not None and cand_chunk_id != tgt_chunk_id:
            return False

        if anchors:
            if isinstance(anchors, str):
                anchors = [anchors]

            compact_text = _compact(cand_text)
            for anchor in anchors:
                if _compact(anchor) not in compact_text:
                    return False

    return True


def first_hit_rank(
    candidates: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    strict: bool = False,
) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    for rank, cand in enumerate(candidates, start=1):
        for tgt in targets:
            if candidate_matches_target(cand, tgt, strict=strict):
                return rank, cand
    return None, None


def evaluate_mode(
    dataset: List[Dict[str, Any]],
    mode: str,
    top_k: int,
) -> Dict[str, Any]:
    n = len(dataset)
    if n == 0:
        raise ValueError("评测集为空")

    doc_hit_1 = 0
    doc_hit_k = 0
    strict_hit_1 = 0
    strict_hit_k = 0

    doc_mrr_k = 0.0
    strict_mrr_k = 0.0

    details = []

    for item in dataset:
        question = item["question"]
        targets = item["targets"]

        candidates = dispatch_retrieval(mode, question, top_k=top_k)

        doc_rank, doc_cand = first_hit_rank(candidates, targets, strict=False)
        strict_rank, strict_cand = first_hit_rank(candidates, targets, strict=True)

        if doc_rank is not None:
            doc_hit_k += 1
            doc_mrr_k += 1.0 / doc_rank
            if doc_rank == 1:
                doc_hit_1 += 1

        if strict_rank is not None:
            strict_hit_k += 1
            strict_mrr_k += 1.0 / strict_rank
            if strict_rank == 1:
                strict_hit_1 += 1

        details.append({
            "id": item["id"],
            "question": question,
            "targets": targets,
            "doc_hit_rank": doc_rank,
            "strict_hit_rank": strict_rank,
            "doc_top1": doc_cand,
            "strict_top1": strict_cand,
            "candidates": candidates[:top_k],
        })

    summary = {
        "mode": mode,
        "top_k": top_k,
        "samples": n,
        "doc_hit@1": round(doc_hit_1 / n, 4),
        "doc_hit@k": round(doc_hit_k / n, 4),
        "strict_hit@1": round(strict_hit_1 / n, 4),
        "strict_hit@k": round(strict_hit_k / n, 4),
        "doc_mrr@k": round(doc_mrr_k / n, 4),
        "strict_mrr@k": round(strict_mrr_k / n, 4),
    }

    return {
        "summary": summary,
        "details": details,
    }


def print_summary_table(results: Dict[str, Dict[str, Any]]) -> None:
    modes = list(results.keys())

    headers = [
        "mode",
        "doc@1",
        "doc@k",
        "strict@1",
        "strict@k",
        "docMRR",
        "strictMRR",
    ]

    rows = []
    for mode in modes:
        s = results[mode]["summary"]
        rows.append([
            s["mode"],
            f'{s["doc_hit@1"]:.4f}',
            f'{s["doc_hit@k"]:.4f}',
            f'{s["strict_hit@1"]:.4f}',
            f'{s["strict_hit@k"]:.4f}',
            f'{s["doc_mrr@k"]:.4f}',
            f'{s["strict_mrr@k"]:.4f}',
        ])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print("\n" + fmt_row(headers))
    print("-|-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))
    print()


def print_single_summary(result: Dict[str, Any]) -> None:
    s = result["summary"]
    print("\n==================== 评测结果 ====================")
    print(f"mode       : {s['mode']}")
    print(f"samples    : {s['samples']}")
    print(f"top_k      : {s['top_k']}")
    print(f"doc_hit@1  : {s['doc_hit@1']:.4f}")
    print(f"doc_hit@k  : {s['doc_hit@k']:.4f}")
    print(f"strict_hit@1: {s['strict_hit@1']:.4f}")
    print(f"strict_hit@k: {s['strict_hit@k']:.4f}")
    print(f"doc_mrr@k  : {s['doc_mrr@k']:.4f}")
    print(f"strict_mrr@k: {s['strict_mrr@k']:.4f}")
    print("==================================================\n")

# 导出summary.csv
def export_summary_csv(results: Dict[str, Dict[str, Any]], path: Path) -> None:
    headers = [
        "mode",
        "samples",
        "top_k",
        "doc_hit@1",
        "doc_hit@k",
        "strict_hit@1",
        "strict_hit@k",
        "doc_mrr@k",
        "strict_mrr@k",
    ]

    rows = []
    for mode, result in results.items():
        s = result["summary"]
        rows.append({
            "mode": s["mode"],
            "samples": s["samples"],
            "top_k": s["top_k"],
            "doc_hit@1": s["doc_hit@1"],
            "doc_hit@k": s["doc_hit@k"],
            "strict_hit@1": s["strict_hit@1"],
            "strict_hit@k": s["strict_hit@k"],
            "doc_mrr@k": s["doc_mrr@k"],
            "strict_mrr@k": s["strict_mrr@k"],
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

def _safe_value(v):
    return "" if v is None else v

# 导出details.csv
def export_details_csv(results: Dict[str, Dict[str, Any]], path: Path) -> None:
    headers = [
        "mode",
        "id",
        "question",
        "targets",
        "doc_hit_rank",
        "strict_hit_rank",

        "doc_hit_source",
        "doc_hit_chunk_id",
        "doc_hit_score",
        "doc_hit_hybrid_score",
        "doc_hit_rerank_score",
        "doc_hit_lexical_score",
        "doc_hit_vector_score",
        "doc_hit_bm25_score",

        "strict_hit_source",
        "strict_hit_chunk_id",
        "strict_hit_score",
        "strict_hit_hybrid_score",
        "strict_hit_rerank_score",
        "strict_hit_lexical_score",
        "strict_hit_vector_score",
        "strict_hit_bm25_score",
    ]

    rows = []

    for mode, result in results.items():
        for item in result["details"]:
            doc_cand = item.get("doc_top1") or {}
            strict_cand = item.get("strict_top1") or {}

            rows.append({
                "mode": mode,
                "id": item.get("id", ""),
                "question": item.get("question", ""),
                "targets": json.dumps(item.get("targets", []), ensure_ascii=False),

                "doc_hit_rank": _safe_value(item.get("doc_hit_rank")),
                "strict_hit_rank": _safe_value(item.get("strict_hit_rank")),

                "doc_hit_source": _safe_value(doc_cand.get("source")),
                "doc_hit_chunk_id": _safe_value(doc_cand.get("chunk_id")),
                "doc_hit_score": _safe_value(doc_cand.get("score")),
                "doc_hit_hybrid_score": _safe_value(doc_cand.get("hybrid_score")),
                "doc_hit_rerank_score": _safe_value(doc_cand.get("rerank_score")),
                "doc_hit_lexical_score": _safe_value(doc_cand.get("lexical_score")),
                "doc_hit_vector_score": _safe_value(doc_cand.get("vector_score")),
                "doc_hit_bm25_score": _safe_value(doc_cand.get("bm25_score")),

                "strict_hit_source": _safe_value(strict_cand.get("source")),
                "strict_hit_chunk_id": _safe_value(strict_cand.get("chunk_id")),
                "strict_hit_score": _safe_value(strict_cand.get("score")),
                "strict_hit_hybrid_score": _safe_value(strict_cand.get("hybrid_score")),
                "strict_hit_rerank_score": _safe_value(strict_cand.get("rerank_score")),
                "strict_hit_lexical_score": _safe_value(strict_cand.get("lexical_score")),
                "strict_hit_vector_score": _safe_value(strict_cand.get("vector_score")),
                "strict_hit_bm25_score": _safe_value(strict_cand.get("bm25_score")),
            })

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

# 导出summary.md
def export_summary_md(
    results: Dict[str, Dict[str, Any]],
    path: Path,
    meta: Dict[str, Any],
) -> None:
    lines = []
    lines.append("# Retrieval Evaluation Report")
    lines.append("")
    lines.append("## Meta")
    lines.append(f"- data: `{meta['data']}`")
    lines.append(f"- top_k: `{meta['top_k']}`")
    lines.append(f"- modes: `{', '.join(meta['modes'])}`")
    lines.append(f"- samples: `{meta['samples']}`")
    lines.append(f"- generated_at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append("")

    lines.append("## Summary")
    headers = [
        "mode",
        "samples",
        "top_k",
        "doc_hit@1",
        "doc_hit@k",
        "strict_hit@1",
        "strict_hit@k",
        "doc_mrr@k",
        "strict_mrr@k",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for mode, result in results.items():
        s = result["summary"]
        row = [
            str(s["mode"]),
            str(s["samples"]),
            str(s["top_k"]),
            f'{s["doc_hit@1"]:.4f}',
            f'{s["doc_hit@k"]:.4f}',
            f'{s["strict_hit@1"]:.4f}',
            f'{s["strict_hit@k"]:.4f}',
            f'{s["doc_mrr@k"]:.4f}',
            f'{s["strict_mrr@k"]:.4f}',
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Notes")
    lines.append("- `summary.csv` 适合 Excel / WPS / 论文表格。")
    lines.append("- `details.csv` 适合逐题排查检索是否命中。")
    lines.append("- 如果当前数据集较小，所有指标接近 1 属于正常现象。")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

# 打包导出函数
def export_reports(
    results: Dict[str, Dict[str, Any]],
    export_dir: Path,
    data_path: Path,
    top_k: int,
    modes: List[str],
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "data": str(data_path),
        "top_k": top_k,
        "modes": modes,
        "samples": len(next(iter(results.values()))["details"]) if results else 0,
    }

    # 原始 JSON
    report_json = {
        "meta": meta,
        "results": results,
    }
    (export_dir / "report.json").write_text(
        json.dumps(report_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # CSV / MD
    export_summary_csv(results, export_dir / "summary.csv")
    export_details_csv(results, export_dir / "details.csv")
    export_summary_md(results, export_dir / "summary.md", meta)

    print(f"[INFO] 已导出评测结果到目录: {export_dir}")
    print(f"[INFO] - {export_dir / 'report.json'}")
    print(f"[INFO] - {export_dir / 'summary.csv'}")
    print(f"[INFO] - {export_dir / 'details.csv'}")
    print(f"[INFO] - {export_dir / 'summary.md'}")
def main():
    parser = argparse.ArgumentParser(description="RAG 检索评测脚本")
    parser.add_argument(
        "--data",
        type=str,
        default="data/eval/retrieval_eval.json",
        help="评测集 JSON 文件路径",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="评测时取前 K 个结果",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="current",
        choices=["current", "hybrid", "vector", "bm25"],
        help="单模式评测",
    )
    parser.add_argument(
        "--compare",
        nargs="*",
        default=None,
        help="对比多个模式，例如: --compare current hybrid vector bm25",
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="保存详细报告到 JSON 文件",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="导出 CSV / Markdown / JSON 报告的目录",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    dataset = load_eval_data(data_path)

    modes = args.compare if args.compare else [args.mode]
    results = {}

    for mode in modes:
        print(f"[INFO] 正在评测 mode={mode} ...")
        results[mode] = evaluate_mode(dataset, mode=mode, top_k=args.top_k)
        if not args.compare:
            print_single_summary(results[mode])

    if args.compare:
        print_summary_table(results)

    if args.save_report:
        report = {
            "meta": {
                "data": str(data_path),
                "top_k": args.top_k,
                "modes": modes,
                "samples": len(dataset),
            },
            "results": results,
        }
    if args.export_dir:
        export_reports(
            results=results,
            export_dir=Path(args.export_dir),
            data_path=data_path,
            top_k=args.top_k,
            modes=modes,
        )
    
        save_path = Path(args.save_report)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] 评测报告已保存到: {save_path}")


if __name__ == "__main__":
    main()