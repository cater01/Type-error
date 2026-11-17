#!/usr/bin/env python3
"""Audit experiment 5/6 JSONL results and export cross-strategy file lists."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any



# Shared CLI/config support; works when invoked from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from runtime import (REPO_ROOT, ConfigArgumentParser, generation_datasets,
                     external_root, configure_java, finish_setup, seed_torch,
                     credential, repo_path)

WORKSPACE = REPO_ROOT
OUTPUT_DIR = REPO_ROOT / 'results/analysis'

MODEL_LABELS = {
    "magicoder": "Magicoder-S-DS-6.7B",
    "qwen7b": "Qwen2.5-Coder-7B",
    "qwen32b": "Qwen2.5-Coder-32B",
}
DATASET_LABELS = {
    "avatar": "AVATAR",
    "codenet": "CodeNet",
    "transcoder": "TransCoder",
    "xcodeeval": "XCodeEval",
}

EXPERIMENTS = {
    "实验5": {
        "root": REPO_ROOT / "results/rq4/tests",
        "universe_root": REPO_ROOT / "results/rq4/generation",
        "universe_filename": "translation_results.jsonl",
        "output_statuses": {"ok"},
        "strategies": ("p0", "p1", "p2"),
        "expected": {
            ("magicoder", "p0"): (517, 429, 41),
            ("magicoder", "p1"): (517, 426, 64),
            ("magicoder", "p2"): (517, 408, 87),
            ("qwen7b", "p0"): (536, 384, 48),
            ("qwen7b", "p1"): (536, 384, 78),
            ("qwen7b", "p2"): (536, 337, 155),
            ("qwen32b", "p0"): (548, 471, 39),
            ("qwen32b", "p1"): (548, 463, 92),
            ("qwen32b", "p2"): (548, 454, 200),
        },
        "overall_expected": {
            "p0": (1601, 1284, 128),
            "p1": (1601, 1273, 234),
            "p2": (1601, 1199, 442),
        },
    },
    "实验6": {
        "root": REPO_ROOT / "results/rq5/tests",
        "universe_root": REPO_ROOT / "results/rq5/generation",
        "universe_filename": "repair_results.jsonl",
        "output_statuses": {"ok", "skipped_existing"},
        "strategies": ("r0", "r1", "r2"),
        "expected": {
            ("magicoder", "r0"): (517, 428, 75),
            ("magicoder", "r1"): (517, 445, 101),
            ("magicoder", "r2"): (517, 429, 129),
            ("qwen7b", "r0"): (536, 428, 163),
            ("qwen7b", "r1"): (536, 408, 194),
            ("qwen7b", "r2"): (536, 370, 214),
            ("qwen32b", "r0"): (548, 520, 248),
            ("qwen32b", "r1"): (548, 521, 293),
            ("qwen32b", "r2"): (548, 516, 309),
        },
        "overall_expected": {
            "r0": (1601, 1376, 486),
            "r1": (1601, 1374, 588),
            "r2": (1601, 1315, 652),
        },
    },
}


@dataclass(frozen=True)
class Result:
    model: str
    strategy: str
    dataset: str
    name: str
    status: str
    passed: bool
    compile_status: str
    java_file: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.model, self.dataset, self.name)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_jsonl(path: Path, model: str, strategy: str, dataset: str) -> list[Result]:
    records: list[Result] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                raw = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON 解析失败：{path}:{line_number}: {exc}") from exc
            nested_compile_status = str((raw.get("compile") or {}).get("status", ""))
            if nested_compile_status:
                compile_status = nested_compile_status
            elif raw.get("compile_returncode") == 0:
                compile_status = "ok"
            elif "compile_returncode" in raw:
                compile_status = "error"
            else:
                compile_status = ""
            records.append(
                Result(
                    model=model,
                    strategy=strategy,
                    dataset=dataset,
                    name=str(raw.get("name", "")),
                    status=str(raw.get("status", "")),
                    passed=raw.get("passed") is True,
                    compile_status=compile_status,
                    java_file=str(raw.get("java_file", "")),
                )
            )
    return records


def discover_and_load(experiment_name: str, config: dict[str, Any]) -> tuple[dict[str, dict[tuple[str, str, str], Result]], list[str], list[dict[str, Any]]]:
    root: Path = config["root"]
    strategies: tuple[str, ...] = config["strategies"]
    if not root.is_dir():
        raise FileNotFoundError(f"结果目录不存在：{root}")

    by_strategy: dict[str, dict[tuple[str, str, str], Result]] = {s: {} for s in strategies}
    problems: list[str] = []
    summary_checks: list[dict[str, Any]] = []
    jsonl_paths = sorted(root.glob("*/*/*/test_results.jsonl"))

    expected_file_count = len(MODEL_LABELS) * len(strategies) * len(DATASET_LABELS)
    if len(jsonl_paths) != expected_file_count:
        problems.append(
            f"{experiment_name}: 预期 {expected_file_count} 个 JSONL，实际发现 {len(jsonl_paths)} 个"
        )

    for path in jsonl_paths:
        relative = path.relative_to(root)
        model, strategy, dataset_dir, _ = relative.parts
        model = model.lower()
        strategy = strategy.lower()
        dataset = dataset_dir.lower()
        if model not in MODEL_LABELS:
            problems.append(f"{experiment_name}: 未识别模型目录 {relative}")
            continue
        if strategy not in strategies:
            problems.append(f"{experiment_name}: 未识别策略目录 {relative}")
            continue
        if dataset not in DATASET_LABELS:
            problems.append(f"{experiment_name}: 未识别数据集目录 {relative}")
            continue

        records = read_jsonl(path, model, strategy, dataset)
        duplicate_names = [name for name, count in Counter(r.name for r in records).items() if count > 1]
        if duplicate_names:
            problems.append(
                f"{experiment_name}/{model}/{strategy}/{dataset}: JSONL 内存在重复 name: {duplicate_names[:5]}"
            )

        for result in records:
            if not result.name:
                problems.append(f"{experiment_name}: 空 name 记录：{path}")
            if result.passed != (result.status == "passed"):
                problems.append(
                    f"{experiment_name}/{model}/{strategy}/{dataset}/{result.name}: passed 与 status 不一致"
                )
            if result.key in by_strategy[strategy]:
                problems.append(f"{experiment_name}/{strategy}: 跨文件重复键 {result.key}")
            by_strategy[strategy][result.key] = result

        summary_path = path.with_name("test_results.summary.json")
        if not summary_path.is_file():
            problems.append(f"{experiment_name}: 缺少 summary：{summary_path}")
            continue
        summary = read_json(summary_path)
        raw_counts = (
            len(records),
            sum(r.compile_status == "ok" for r in records),
            sum(r.passed for r in records),
        )
        summary_counts = (
            int(summary.get("total_files", 0)),
            int(summary.get("total_files", 0)) - int(summary.get("compile_errors", 0)),
            int(summary.get("passed_files", 0)),
        )
        summary_checks.append(
            {
                "experiment": experiment_name,
                "model": model,
                "strategy": strategy,
                "dataset": dataset,
                "raw_counts": raw_counts,
                "summary_counts": summary_counts,
                "match": raw_counts == summary_counts,
            }
        )
        if raw_counts != summary_counts:
            problems.append(
                f"{experiment_name}/{model}/{strategy}/{dataset}: 原始 JSONL {raw_counts} != summary {summary_counts}"
            )

    base_keys = set(by_strategy[strategies[0]])
    for strategy in strategies[1:]:
        current_keys = set(by_strategy[strategy])
        missing = sorted(base_keys - current_keys)
        extra = sorted(current_keys - base_keys)
        if missing or extra:
            problems.append(
                f"{experiment_name}/{strategy}: 与 {strategies[0]} 的样本集合不同；缺少 {len(missing)}，新增 {len(extra)}"
            )

    return by_strategy, problems, summary_checks


def load_universe(
    experiment_name: str,
    config: dict[str, Any],
    by_strategy: dict[str, dict[tuple[str, str, str], Result]],
) -> tuple[dict[tuple[str, str], int], list[dict[str, Any]], list[str]]:
    universe_counts: dict[tuple[str, str], int] = {}
    non_output_rows: list[dict[str, Any]] = []
    problems: list[str] = []
    output_statuses: set[str] = config["output_statuses"]
    for model in MODEL_LABELS:
        for strategy in config["strategies"]:
            path = config["universe_root"] / model / strategy / config["universe_filename"]
            if not path.is_file():
                problems.append(f"{experiment_name}/{model}/{strategy}: 缺少任务全集记录 {path}")
                continue
            items: list[dict[str, Any]] = []
            with path.open("r", encoding="utf-8-sig") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    if not raw_line.strip():
                        continue
                    try:
                        items.append(json.loads(raw_line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"JSON 解析失败：{path}:{line_number}: {exc}") from exc
            universe_counts[(model, strategy)] = len(items)
            output_items = [item for item in items if str(item.get("status", "")) in output_statuses]
            tested_records = len(by_strategy[strategy])
            tested_model_records = sum(r.model == model for r in by_strategy[strategy].values())
            if len(output_items) != tested_model_records:
                problems.append(
                    f"{experiment_name}/{model}/{strategy}: 产生输出 {len(output_items)} 条，但测试 JSONL 有 {tested_model_records} 条"
                )
            for item in items:
                status = str(item.get("status", ""))
                if status not in output_statuses:
                    non_output_rows.append(
                        {
                            "experiment": experiment_name,
                            "model": MODEL_LABELS[model],
                            "strategy": strategy.upper(),
                            "source_file": str(item.get("source_file", "")),
                            "status": status,
                        }
                    )
    return universe_counts, non_output_rows, problems


def make_audit_rows(
    experiment_name: str,
    config: dict[str, Any],
    by_strategy: dict[str, dict[tuple[str, str, str], Result]],
    universe_counts: dict[tuple[str, str], int],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    expected: dict[tuple[str, str], tuple[int, int, int]] = config["expected"]

    for strategy in config["strategies"]:
        for model in MODEL_LABELS:
            model_records = [r for r in by_strategy[strategy].values() if r.model == model]
            actual = (
                universe_counts[(model, strategy)],
                sum(r.compile_status == "ok" for r in model_records),
                sum(r.passed for r in model_records),
            )
            target = expected[(model, strategy)]
            match = actual == target
            if not match:
                problems.append(
                    f"{experiment_name}/{model}/{strategy}: 复算 {actual} != 表格 {target}"
                )
            rows.append(
                audit_row(
                    experiment_name,
                    MODEL_LABELS[model],
                    strategy,
                    actual,
                    target,
                    match,
                    tested_records=len(model_records),
                )
            )

        all_records = list(by_strategy[strategy].values())
        actual = (
            sum(universe_counts[(model, strategy)] for model in MODEL_LABELS),
            sum(r.compile_status == "ok" for r in all_records),
            sum(r.passed for r in all_records),
        )
        target = config["overall_expected"][strategy]
        match = actual == target
        if not match:
            problems.append(f"{experiment_name}/overall/{strategy}: 复算 {actual} != 表格 {target}")
        rows.append(
            audit_row(
                experiment_name,
                "Overall",
                strategy,
                actual,
                target,
                match,
                tested_records=len(all_records),
            )
        )

    return rows, problems


def audit_row(
    experiment: str,
    model: str,
    strategy: str,
    actual: tuple[int, int, int],
    expected: tuple[int, int, int],
    match: bool,
    tested_records: int,
) -> dict[str, Any]:
    total, compile_pass, test_pass = actual
    return {
        "experiment": experiment,
        "model": model,
        "strategy": strategy.upper(),
        "total": total,
        "tested_records": tested_records,
        "not_tested_records": total - tested_records,
        "compile_pass": compile_pass,
        "compile_rate": f"{compile_pass / total:.2%}" if total else "",
        "test_pass": test_pass,
        "test_rate": f"{test_pass / total:.2%}" if total else "",
        "expected_total": expected[0],
        "expected_compile_pass": expected[1],
        "expected_test_pass": expected[2],
        "matches_table": match,
    }


def comparison_row(
    key: tuple[str, str, str],
    records: dict[str, dict[tuple[str, str, str], Result]],
    strategies: tuple[str, str, str],
) -> dict[str, Any]:
    model, dataset, name = key
    row: dict[str, Any] = {
        "model": MODEL_LABELS[model],
        "dataset": DATASET_LABELS[dataset],
        "file_name": name,
    }
    for strategy in strategies:
        result = records[strategy][key]
        label = strategy.upper()
        row[f"{label}_passed"] = result.passed
        row[f"{label}_status"] = result.status
        row[f"{label}_compile_status"] = result.compile_status
        row[f"{label}_java_file"] = result.java_file
    return row


def filter_lists(
    records: dict[str, dict[tuple[str, str, str], Result]],
    strategies: tuple[str, str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    s0, s1, s2 = strategies
    regression_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    for key in sorted(records[s0]):
        r0, r1, r2 = records[s0][key], records[s1][key], records[s2][key]
        if r0.passed and (not r1.passed or not r2.passed):
            row = comparison_row(key, records, strategies)
            row["failure_pattern"] = (
                f"{s1.upper()}和{s2.upper()}均失败"
                if not r1.passed and not r2.passed
                else f"仅{s1.upper()}失败"
                if not r1.passed
                else f"仅{s2.upper()}失败"
            )
            regression_rows.append(row)
        if not r0.passed and not r1.passed and r2.passed:
            recovery_rows.append(comparison_row(key, records, strategies))
    return regression_rows, recovery_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"没有可写入的行：{path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pattern_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(row["failure_pattern"]) for row in rows)


def grouped_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter((str(row["model"]), str(row["dataset"])) for row in rows)
    return [
        {"model": model, "dataset": dataset, "count": count}
        for (model, dataset), count in sorted(counts.items())
    ]


def markdown_audit_table(rows: list[dict[str, Any]], experiment: str) -> list[str]:
    selected = [row for row in rows if row["experiment"] == experiment]
    lines = [
        "| Model | Strategy | Total | Compile Pass | Compile Rate | Test Pass | Test Rate | Match |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['model']} | {row['strategy']} | {row['total']} | {row['compile_pass']} | "
            f"{row['compile_rate']} | {row['test_pass']} | {row['test_rate']} | "
            f"{'是' if row['matches_table'] else '否'} |"
        )
    return lines


def main() -> int:
    global OUTPUT_DIR
    parser = ConfigArgumentParser(description='Audit prevention/repair results against task universes and recorded tables.')
    parser.add_argument('--prevention-results', type=Path, default=REPO_ROOT / 'results/rq4/tests')
    parser.add_argument('--prevention-generation', type=Path, default=REPO_ROOT / 'results/rq4/generation')
    parser.add_argument('--repair-results', type=Path, default=REPO_ROOT / 'results/rq5/tests')
    parser.add_argument('--repair-generation', type=Path, default=REPO_ROOT / 'results/rq5/generation')
    parser.add_argument('--out-dir', type=Path, default=REPO_ROOT / 'results/analysis')
    args = parser.parse_args()
    OUTPUT_DIR = args.out_dir
    EXPERIMENTS['实验5']['root'], EXPERIMENTS['实验5']['universe_root'] = args.prevention_results, args.prevention_generation
    EXPERIMENTS['实验6']['root'], EXPERIMENTS['实验6']['universe_root'] = args.repair_results, args.repair_generation
    finish_setup(args, OUTPUT_DIR / 'run_config.json')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_audit_rows: list[dict[str, Any]] = []
    all_problems: list[str] = []
    all_summary_checks: list[dict[str, Any]] = []
    all_non_output_rows: list[dict[str, Any]] = []
    loaded: dict[str, dict[str, dict[tuple[str, str, str], Result]]] = {}

    for experiment_name, config in EXPERIMENTS.items():
        by_strategy, load_problems, summary_checks = discover_and_load(experiment_name, config)
        loaded[experiment_name] = by_strategy
        all_problems.extend(load_problems)
        all_summary_checks.extend(summary_checks)
        universe_counts, non_output_rows, universe_problems = load_universe(
            experiment_name, config, by_strategy
        )
        all_non_output_rows.extend(non_output_rows)
        all_problems.extend(universe_problems)
        audit_rows, audit_problems = make_audit_rows(
            experiment_name, config, by_strategy, universe_counts
        )
        all_audit_rows.extend(audit_rows)
        all_problems.extend(audit_problems)

    exp5_regression, exp5_recovery = filter_lists(loaded["实验5"], ("p0", "p1", "p2"))
    exp6_regression, exp6_recovery = filter_lists(loaded["实验6"], ("r0", "r1", "r2"))

    outputs = {
        "01_P0正确但P1或P2失败.csv": exp5_regression,
        "02_R0正确但R1或R2失败.csv": exp6_regression,
        "03_P0与P1失败但P2正确.csv": exp5_recovery,
        "04_R0与R1失败但R2正确.csv": exp6_recovery,
        "05_四类名单按模型与数据集计数.csv": (
            [dict(list(row.items()) + [("list", "P0正确但P1或P2失败")]) for row in grouped_counts(exp5_regression)]
            + [dict(list(row.items()) + [("list", "R0正确但R1或R2失败")]) for row in grouped_counts(exp6_regression)]
            + [dict(list(row.items()) + [("list", "P0与P1失败但P2正确")]) for row in grouped_counts(exp5_recovery)]
            + [dict(list(row.items()) + [("list", "R0与R1失败但R2正确")]) for row in grouped_counts(exp6_recovery)]
        ),
        "06_未产生可测试代码的记录.csv": all_non_output_rows,
    }

    # Put the list label first in the compact count file.
    count_rows = outputs["05_四类名单按模型与数据集计数.csv"]
    outputs["05_四类名单按模型与数据集计数.csv"] = [
        {"list": row["list"], "model": row["model"], "dataset": row["dataset"], "count": row["count"]}
        for row in count_rows
    ]

    write_csv(OUTPUT_DIR / "00_正确率核对.csv", all_audit_rows)
    for filename, rows in outputs.items():
        write_csv(OUTPUT_DIR / filename, rows)

    audit_json = {
        "all_counts_match_tables": all(row["matches_table"] for row in all_audit_rows),
        "all_raw_jsonl_match_summaries": all(check["match"] for check in all_summary_checks),
        "all_strategy_sample_sets_aligned": not any("样本集合不同" in problem for problem in all_problems),
        "non_output_record_count": len(all_non_output_rows),
        "problems": all_problems,
        "summary_checks": all_summary_checks,
        "list_counts": {
            "P0正确但P1或P2失败": len(exp5_regression),
            "R0正确但R1或R2失败": len(exp6_regression),
            "P0与P1失败但P2正确": len(exp5_recovery),
            "R0与R1失败但R2正确": len(exp6_recovery),
        },
        "failure_patterns": {
            "P0正确但P1或P2失败": pattern_counts(exp5_regression),
            "R0正确但R1或R2失败": pattern_counts(exp6_regression),
        },
    }
    with (OUTPUT_DIR / "核对详情.json").open("w", encoding="utf-8") as handle:
        json.dump(audit_json, handle, ensure_ascii=False, indent=2)

    md_lines = [
        "# 实验 5 与实验 6 结果核对及筛选摘要",
        "",
        "判定口径：嵌套 `compile.status == \"ok\"` 或 TransCoder 的 `compile_returncode == 0` 视为编译通过；文件级 `passed == true` 视为翻译正确。表格 Total 来自翻译/修复任务全集，未产生可测试 Java 代码的记录按失败计入分母。",
        "",
        "## 第一阶段：正确率核对",
        "",
        f"- 原始 JSONL 与各目录 summary：{'全部一致' if audit_json['all_raw_jsonl_match_summaries'] else '存在不一致'}",
        f"- 逐策略样本集合：{'完全对齐' if audit_json['all_strategy_sample_sets_aligned'] else '存在不一致'}",
        f"- 与截图表 20–23：{'全部一致' if audit_json['all_counts_match_tables'] else '存在不一致'}",
        f"- 未产生可测试代码、因此未进入测试 JSONL 的记录：{len(all_non_output_rows)} 条（跨策略计数）",
        "",
        "### 实验 5（P0/P1/P2）",
        "",
        *markdown_audit_table(all_audit_rows, "实验5"),
        "",
        "### 实验 6（R0/R1/R2）",
        "",
        *markdown_audit_table(all_audit_rows, "实验6"),
        "",
        "## 第二、三阶段：名单数量",
        "",
        f"- P0 正确但 P1 或 P2 失败：{len(exp5_regression)}",
        f"  - {dict(pattern_counts(exp5_regression))}",
        f"- R0 正确但 R1 或 R2 失败：{len(exp6_regression)}",
        f"  - {dict(pattern_counts(exp6_regression))}",
        f"- P0、P1 均失败但 P2 正确：{len(exp5_recovery)}",
        f"- R0、R1 均失败但 R2 正确：{len(exp6_recovery)}",
        "",
        "完整名单见同目录 CSV 文件。CSV 使用 UTF-8 BOM，可直接用 Excel 打开。",
    ]
    if all_problems:
        md_lines.extend(["", "## 数据问题", "", *[f"- {problem}" for problem in all_problems]])
    (OUTPUT_DIR / "README_结果核对与筛选摘要.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps(audit_json, ensure_ascii=False, indent=2, default=dict))
    return 1 if all_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
