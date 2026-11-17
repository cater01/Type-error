import json
from pathlib import Path


# =========================
# 1. 路径配置
# =========================


# Shared CLI/config support; works when invoked from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from runtime import (REPO_ROOT, ConfigArgumentParser, generation_datasets,
                     external_root, configure_java, finish_setup, seed_torch,
                     credential, repo_path)

CLAUDE_FILE = None

GEMINI_FILE = None

OUTPUT_DIR = None

# 输出文件
BOTH_TYPE_FILE = None
BOTH_NON_TYPE_FILE = None
DISAGREEMENT_FILE = None
OTHER_FILE = None
ALL_MERGED_FILE = None


LABELS = {"type error", "non-type error"}


# =========================
# 2. 工具函数
# =========================

def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} 第 {line_num} 行 JSON 解析失败: {e}")

    return rows


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_key(row: dict) -> str:
    """
    默认按 id 匹配。
    如果你担心不同文件里 id 重复，可以改成：
    return f"{row.get('id', '')}||{row.get('source_file', '')}||{row.get('target_file', '')}"
    """
    return str(row.get("id", "")).strip()


def index_by_key(rows: list[dict], model_name: str) -> dict[str, dict]:
    indexed = {}
    duplicated = []

    for row in rows:
        key = make_key(row)

        if not key:
            print(f"警告：{model_name} 中存在没有 id 的样本，已跳过：{row}")
            continue

        if key in indexed:
            duplicated.append(key)

        indexed[key] = row

    if duplicated:
        print(f"警告：{model_name} 中存在重复 id，共 {len(duplicated)} 个。示例：{duplicated[:5]}")
        print("当前脚本会保留最后一次出现的记录。")

    return indexed


# =========================
# 3. 主流程
# =========================

def main():
    global CLAUDE_FILE, GEMINI_FILE, OUTPUT_DIR, BOTH_TYPE_FILE, BOTH_NON_TYPE_FILE, DISAGREEMENT_FILE, OTHER_FILE, ALL_MERGED_FILE
    parser = ConfigArgumentParser(description='Align and group dual-model predictions.')
    parser.add_argument('--claude-file', type=Path, required=True)
    parser.add_argument('--gemini-file', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    CLAUDE_FILE, GEMINI_FILE, OUTPUT_DIR = args.claude_file, args.gemini_file, args.out_dir
    BOTH_TYPE_FILE = OUTPUT_DIR / 'both_type_error.jsonl'
    BOTH_NON_TYPE_FILE = OUTPUT_DIR / 'both_non_type_error.jsonl'
    DISAGREEMENT_FILE = OUTPUT_DIR / 'disagreement.jsonl'
    OTHER_FILE = OUTPUT_DIR / 'other_invalid_or_missing.jsonl'
    ALL_MERGED_FILE = OUTPUT_DIR / 'all_merged_predictions.jsonl'
    finish_setup(args, OUTPUT_DIR / 'run_config.json')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    claude_rows = load_jsonl(CLAUDE_FILE)
    gemini_rows = load_jsonl(GEMINI_FILE)

    print(f"Claude 结果数: {len(claude_rows)}")
    print(f"Gemini 结果数: {len(gemini_rows)}")

    claude_map = index_by_key(claude_rows, "Claude")
    gemini_map = index_by_key(gemini_rows, "Gemini")

    all_keys = sorted(set(claude_map.keys()) | set(gemini_map.keys()), key=lambda x: int(x) if x.isdigit() else x)

    both_type = []
    both_non_type = []
    disagreement = []
    other = []
    all_merged = []

    for key in all_keys:
        c = claude_map.get(key)
        g = gemini_map.get(key)

        if c is None or g is None:
            merged = {
                "id": key,
                "source_file": (c or g or {}).get("source_file", ""),
                "target_file": (c or g or {}).get("target_file", ""),
                "claude_pred": c.get("pred", "") if c else "MISSING",
                "gemini_pred": g.get("pred", "") if g else "MISSING",
                "agreement": False,
                "final_group": "missing",
                "claude_raw_output": c.get("raw_output", "") if c else "",
                "gemini_raw_output": g.get("raw_output", "") if g else "",
                "claude_reason": c.get("reason", "") if c else "",
                "gemini_reason": g.get("reason", "") if g else "",
            }

            other.append(merged)
            all_merged.append(merged)
            continue

        claude_pred = c.get("pred", "")
        gemini_pred = g.get("pred", "")

        merged = {
            "id": key,
            "source_file": c.get("source_file", "") or g.get("source_file", ""),
            "target_file": c.get("target_file", "") or g.get("target_file", ""),

            "claude_model": c.get("model", ""),
            "gemini_model": g.get("model", ""),

            "claude_pred": claude_pred,
            "gemini_pred": gemini_pred,

            "claude_reason": c.get("reason", ""),
            "gemini_reason": g.get("reason", ""),

            "claude_raw_output": c.get("raw_output", ""),
            "gemini_raw_output": g.get("raw_output", ""),
        }

        if claude_pred not in LABELS or gemini_pred not in LABELS:
            merged["agreement"] = False
            merged["final_group"] = "invalid"
            other.append(merged)

        elif claude_pred == gemini_pred == "type error":
            merged["agreement"] = True
            merged["auto_label"] = "type error"
            merged["confidence"] = "high"
            merged["final_group"] = "both_type_error"
            both_type.append(merged)

        elif claude_pred == gemini_pred == "non-type error":
            merged["agreement"] = True
            merged["auto_label"] = "non-type error"
            merged["confidence"] = "high"
            merged["final_group"] = "both_non_type_error"
            both_non_type.append(merged)

        else:
            merged["agreement"] = False
            merged["auto_label"] = ""
            merged["confidence"] = "low"
            merged["final_group"] = "disagreement"
            disagreement.append(merged)

        all_merged.append(merged)

    save_jsonl(both_type, BOTH_TYPE_FILE)
    save_jsonl(both_non_type, BOTH_NON_TYPE_FILE)
    save_jsonl(disagreement, DISAGREEMENT_FILE)
    save_jsonl(other, OTHER_FILE)
    save_jsonl(all_merged, ALL_MERGED_FILE)

    print("\n===== 合并完成 =====")
    print(f"总样本数: {len(all_merged)}")
    print(f"两个模型都预测 type error: {len(both_type)}")
    print(f"两个模型都预测 non-type error: {len(both_non_type)}")
    print(f"两个模型预测不一致: {len(disagreement)}")
    print(f"缺失 / INVALID / 其他异常: {len(other)}")

    print("\n输出文件:")
    print(f"both type error: {BOTH_TYPE_FILE}")
    print(f"both non-type error: {BOTH_NON_TYPE_FILE}")
    print(f"disagreement: {DISAGREEMENT_FILE}")
    print(f"other: {OTHER_FILE}")
    print(f"all merged: {ALL_MERGED_FILE}")


if __name__ == "__main__":
    main()
