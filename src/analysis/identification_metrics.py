import json
from pathlib import Path

# ========= 配置 =========

# Shared CLI/config support; works when invoked from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from runtime import (REPO_ROOT, ConfigArgumentParser, generation_datasets,
                     external_root, configure_java, finish_setup, seed_torch,
                     credential, repo_path)

INPUT_FILE = None

LABELS = ["type error", "non-type error"]
MODE_FALLBACK = "unknown"


def load_jsonl(path: Path):
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


def main():
    global INPUT_FILE
    parser = ConfigArgumentParser(description='Calculate binary identification metrics from gold/pred JSONL.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=None, help='Optional text report output.')
    args = parser.parse_args()
    INPUT_FILE = args.input
    finish_setup(args, args.out.with_suffix('.run.json') if args.out else None)
    if args.out:
        import contextlib
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('w', encoding='utf-8') as handle, contextlib.redirect_stdout(handle):
            return calculate_metrics()
    return calculate_metrics()

def calculate_metrics():
    from sklearn.metrics import accuracy_score, recall_score, classification_report
    rows = load_jsonl(INPUT_FILE)
    if not rows:
        print("文件为空，没有可评测数据。")
        return

    mode = rows[0].get("mode", MODE_FALLBACK)

    valid_rows = [
        r for r in rows
        if r.get("gold") in LABELS and r.get("pred") in LABELS
    ]
    invalid_count = len(rows) - len(valid_rows)

    print("===== Evaluation =====")
    print(f"Mode: {mode}")
    print(f"Valid samples: {len(valid_rows)}")
    print(f"Invalid outputs: {invalid_count}")

    if not valid_rows:
        print("没有可评测的有效样本，请检查 gold 和 pred 字段。")
        return

    golds = [r["gold"] for r in valid_rows]
    preds = [r["pred"] for r in valid_rows]

    acc = accuracy_score(golds, preds)
    binary_recall = recall_score(golds, preds, pos_label="type error")

    print(f"Accuracy: {acc}")
    print(f"Binary Recall: {binary_recall}")
    print(
        classification_report(
            golds,
            preds,
            labels=["non-type error", "type error"],
            digits=4,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()
