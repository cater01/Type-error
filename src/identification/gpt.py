import json
import time
import re
from pathlib import Path



# =========================
# 0. 基本配置
# =========================


# Shared CLI/config support; works when invoked from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from runtime import (REPO_ROOT, ConfigArgumentParser, generation_datasets,
                     external_root, configure_java, finish_setup, seed_torch,
                     credential, repo_path)

API_KEY = ''

# 改成你要用的 GPT 模型名
# 例如：
# MODEL_NAME = "gpt-5.4"
# MODEL_NAME = "gpt-5.5"
MODEL_NAME = "gpt-5.4"

BASE_URL = "https://yunwu.ai/v1/chat/completions"

TEST_FILE = None

OUTPUT_FILE = None

# 可选模式：
# "zero_shot"
# "define_only"
# "few_shot_only"
# "define_and_few_shot"
# "define_and_few_shot_and_error_message"
MODE = "zero_shot"

# 是否输出判断理由
OUTPUT_REASON = False

# GPT Prompt Caching 是自动的，不需要 cache_control
# 但需要固定 prompt 前缀完全一致，且长度通常 >= 1024 tokens 才可能命中
PRINT_USAGE = True

REQUEST_TIMEOUT = 240
MAX_RETRIES = 4
SLEEP_SECONDS = 3

# 中途保存 + 断点续跑
SAVE_EVERY_RESULT = True
RESUME_FROM_OUTPUT = True

LABELS = [
    "type error",
    "non-type error",
]


# =========================
# 1. few-shot 示例
# =========================
# 这里直接粘贴你原来的 FEW_SHOT_EXAMPLES，不要改内容。
# 我这里只放结构提醒，实际运行时需要替换成你原文件里的完整列表。

FEW_SHOT_EXAMPLES = [
    {
        "id": "3",
        "source_file": "00e90909a77ce9e22bb7cbf1285b0609_2.py",
        "target_file": "00e90909a77ce9e22bb7cbf1285b0609_2.java",
        "source_code": "\n\nk,p=map(int,input().split())\ns=0\nfor i in range(1,k+1):\n    s+=int(str(i)+str(i)[::-1])\nprint(s%p)\n",
        "translated_code": "import java.util.*;\n\npublic class Main {\n    public static void main(String[] args) {\n        Scanner scanner = new Scanner(System.in);\n        int k = scanner.nextInt();\n        int p = scanner.nextInt();\n        int s = 0;\n        for (int i = 1; i <= k; i++) {\n            s += Integer.parseInt(String.valueOf(i) + new StringBuilder().append(i).reverse().toString());\n        }\n        System.out.println(s % p);\n    }\n}",
        "label_0": "type error",
        "note": "Type error: attempted to convert a string to an int, but the value was too large and exceeded the allowable range.",
        "Error message": "RUNTIME_ERROR: Exception in thread \"main\" java.lang.NumberFormatException: For input string: \"2147557412\"",
    },
    {
        "id": "60",
        "source_file": "COMPUTE_AVERAGE_TWO_NUMBERS_WITHOUT_OVERFLOW_1.py",
        "target_file": "COMPUTE_AVERAGE_TWO_NUMBERS_WITHOUT_OVERFLOW_1.java",
        "source_code": "def compute_average ( a , b ) :\n    return ( a // 2 ) + ( b // 2 ) + ( ( a % 2 + b % 2 ) // 2 )",
        "translated_code": "public class Main {\n    public static void main(String[] args) {\n        System.out.println(computeAverage(5, 7));\n    }\n\n    public static double computeAverage(int a, int b) {\n        return (a / 2.0) + (b / 2.0) + ((a % 2 + b % 2) / 2.0);\n    }\n}",
        "label_0": "type error",
        "note": "Type error:The return type of `f_gold` is `int`, while the return type of `f_filled` is `double`. The types are incompatible.",
        "Error message": "OUTPUT_MISMATCH: #Results:1, 10",
    },
    {
        "id": "18",
        "source_file": "0ddc839e17dee20e1a954c1289de7fbd_2.py",
        "target_file": "0ddc839e17dee20e1a954c1289de7fbd_2.java",
        "source_code": "g, v = [input() for i in range(8)], [8, 8]\nfor i in range(8):\n    for j in range(8):\n        if g[i][j] == 'W' and all(g[k][j] == '.' for k in range(i)):\n            v[0] = min(v[0], i)\n        elif g[i][j] == 'B' and all(g[k][j] == '.' for k in range(i + 1, 8)):\n            v[1] = min(v[1], 7 - i)\nprint('A' if v[0] <= v[1] else 'B')",
        "translated_code": "import java.util.Scanner;\n\npublic class Main {\n    public static void main(String[] args) {\n        Scanner scanner = new Scanner(System.in);\n        char[][] board = new char[8][8];\n        int[] whiteRow = {8, 8};\n        int[] blackRow = {8, 8};\n\n        for (int i = 0; i < 8; i++) {\n            String line = scanner.nextLine();\n            for (int j = 0; j < 8; j++) {\n                board[i][j] = line.charAt(j);\n            }\n        }\n\n        for (int i = 0; i < 8; i++) {\n            for (int j = 0; j < 8; j++) {\n                if (board[i][j] == 'W' && isClearPath(board, i, j)) {\n                    whiteRow[0] = Math.min(whiteRow[0], i);\n                } else if (board[i][j] == 'B' && isClearPath(board, i, j)) {\n                    blackRow[0] = Math.min(blackRow[0], 7 - i);\n                }\n            }\n        }\n\n        System.out.println(whiteRow[0] <= blackRow[0]? \"A\" : \"B\");\n    }\n\n    private static boolean isClearPath(char[][] board, int i, int j) {\n        for (int k = 0; k < i; k++) {\n            if (board[k][j]!= '.') {\n                return false;\n            }\n        }\n        return true;\n    }\n}",
        "label_0": "non-type error",
        "note": "Non-type error:There is a serious error in the logic for checking the black piece (`'B'`) path, resulting in inconsistent behavior.",
        "Error message": "OUTPUT_MISMATCH: exp_any='B', got='A'",
    },
]


# =========================
# 2. Prompt 配置
# =========================

ZERO_SHOT_SYSTEM_PROMPT = """
You are an expert in code translation error analysis.

Your task is to classify whether the Java translation error is a type error or a non-type error.
""".strip()


DEFINE_SYSTEM_PROMPT = """
You are an expert in code translation error analysis.

Your task is to classify whether the Java translation error is a type error or a non-type error.

Definitions:
- type error: Type Error refers to cases where the code is not correctly translated due to the use of an inappropriate or incorrect data type; using a more suitable or correct data type would avoid the error.

Specifically, it includes:
1. The translated code uses a data type whose precision or numeric range is insufficient, leading to incorrect results during execution.
2. The types of variables, function parameters, or return values in the translated code are inconsistent with the declared or expected types in the target language, causing the code to fail to compile correctly.
3. The translated code performs an unsupported or incompatible operation on a value of a certain type.

- non-type error: the main problem is not caused by the use of an inappropriate data type, but by other translation issues such as logic, algorithm, control flow, condition checking, indexing, data handling, or output-related mistakes.

Rule:
If the Java error message indicates that a numeric string cannot be parsed by Integer.parseInt, Long.parseLong, or a similar function because the value is too large for the target numeric type, classify it as a type error rather than a non-type error.
""".strip()


def get_output_instruction() -> str:
    if OUTPUT_REASON:
        return (
            "You must follow this exact format.\n"
            "The first line must be exactly one of:\n"
            "Label: type error\n"
            "Label: non-type error\n"
            "The second line must start with:\n"
            "Reason: <brief explanation>\n"
            "Do not output any text before the Label line."
        )

    return (
        "Return exactly one line and nothing else:\n"
        "type error\n"
        "or\n"
        "non-type error"
    )


def get_system_prompt(mode: str) -> str:
    if mode == "zero_shot":
        return ZERO_SHOT_SYSTEM_PROMPT
    if mode == "define_only":
        return DEFINE_SYSTEM_PROMPT
    if mode == "few_shot_only":
        return ZERO_SHOT_SYSTEM_PROMPT
    if mode == "define_and_few_shot":
        return DEFINE_SYSTEM_PROMPT
    if mode == "define_and_few_shot_and_error_message":
        return DEFINE_SYSTEM_PROMPT

    raise ValueError(f"未知 MODE: {mode}")


def get_active_few_shot_examples(mode: str):
    if mode in {
        "few_shot_only",
        "define_and_few_shot",
        "define_and_few_shot_and_error_message",
    }:
        return FEW_SHOT_EXAMPLES
    return []


# =========================
# 3. 构建 Prompt：固定前缀 + 动态样本
# =========================

def build_static_user_prefix(mode: str) -> str:
    """
    这个 user prefix 是每条请求完全相同的部分。
    GPT prompt caching 会自动尝试缓存相同前缀。
    """
    active_examples = get_active_few_shot_examples(mode)
    use_error_message = (mode == "define_and_few_shot_and_error_message")

    parts = []

    parts.append(
        "Given Python source code, translated Java code, and possibly an execution/compiler error message, "
        "classify whether the Java translation error is a type error or a non-type error."
    )
    parts.append("")

    if use_error_message:
        parts.append(
            "You may use the error message as additional evidence, but classify based on the root cause "
            "of the translation error rather than matching surface keywords only."
        )
        parts.append("")

    parts.append(get_output_instruction())
    parts.append("")

    if active_examples:
        parts.append("Few-shot examples:")
        parts.append("")

        for i, ex in enumerate(active_examples, start=1):
            parts.append(f"Example {i}:")
            parts.append("Python:")
            parts.append(ex["source_code"])
            parts.append("")

            parts.append("Java:")
            parts.append(ex["translated_code"])
            parts.append("")

            if use_error_message:
                parts.append("Error message:")
                parts.append(ex.get("Error message", ""))
                parts.append("")

            parts.append(f"Label: {ex['label_0']}")

            if ex.get("note", "").strip():
                parts.append(f"Reason: {ex['note'].strip()}")

            parts.append("")

    parts.append("Now classify the following sample.")
    parts.append("")

    return "\n".join(parts)


def build_dynamic_sample_prompt(sample: dict, mode: str) -> str:
    use_error_message = (mode == "define_and_few_shot_and_error_message")

    parts = []

    parts.append("Python:")
    parts.append(sample["source_code"])
    parts.append("")

    parts.append("Java:")
    parts.append(sample["translated_code"])
    parts.append("")

    if use_error_message:
        parts.append("Error message:")
        parts.append(sample.get("Error message", ""))
        parts.append("")

    parts.append("Now output the answer.")
    parts.append("The first line must be exactly one of:")
    parts.append("Label: type error")
    parts.append("Label: non-type error")

    if OUTPUT_REASON:
        parts.append("The second line must start with:")
        parts.append("Reason: ")

    return "\n".join(parts)


def build_messages(static_user_prefix: str, sample: dict, mode: str) -> list[dict]:
    """
    GPT 自动缓存依赖相同的 prompt 前缀。
    因此 system message 和 user message 的前半部分要稳定不变。
    """
    dynamic_prompt = build_dynamic_sample_prompt(sample, mode)

    full_user_prompt = static_user_prefix + dynamic_prompt

    return [
        {
            "role": "system",
            "content": get_system_prompt(mode),
        },
        {
            "role": "user",
            "content": full_user_prompt,
        },
    ]


# =========================
# 4. 文件工具函数
# =========================

def load_jsonl(path: Path):
    rows = []

    if not path.exists():
        return rows

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


def append_jsonl(row: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_gold_label(sample: dict) -> str:
    return (
        sample.get("label_0", "")
        or sample.get("gold", "")
        or sample.get("label", "")
    ).strip()


def get_sample_key(sample: dict) -> str:
    """
    用于断点续跑。
    优先 id；如果 id 为空，用 source_file + target_file。
    """
    sample_id = str(sample.get("id", "")).strip()

    if sample_id:
        return sample_id

    return f"{sample.get('source_file', '')}||{sample.get('target_file', '')}"


def load_existing_results(path: Path) -> dict[str, dict]:
    existing = {}

    if not path.exists():
        return existing

    rows = load_jsonl(path)

    for row in rows:
        key = get_sample_key(row)
        if key:
            existing[key] = row

    return existing


# =========================
# 5. 校验和解析函数
# =========================

def validate_mode(mode: str):
    allowed = {
        "zero_shot",
        "define_only",
        "few_shot_only",
        "define_and_few_shot",
        "define_and_few_shot_and_error_message",
    }

    if mode not in allowed:
        raise ValueError(f"MODE 必须是 {allowed} 之一，当前是: {mode}")


def check_few_shot_examples():
    if not FEW_SHOT_EXAMPLES:
        return

    for ex in FEW_SHOT_EXAMPLES:
        if ex["label_0"] not in LABELS:
            raise ValueError(
                f"few-shot 示例 {ex['id']} 的 label_0 不合法: {ex['label_0']}"
            )


def normalize_text(text: str) -> str:
    s = text.strip().lower()

    s = s.replace("\u2011", "-")
    s = s.replace("\u2012", "-")
    s = s.replace("\u2013", "-")
    s = s.replace("\u2014", "-")

    s = re.sub(r"[*_`#>\[\]\{\}\(\)]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def extract_label(text: str):
    if not text or not text.strip():
        return "INVALID"

    s = normalize_text(text)

    # 兼容极短输出
    if s in {"non", "non-", "non-type", "non type", "non type error"}:
        return "non-type error"

    if s in {"type", "type-error"}:
        return "type error"

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first_line = normalize_text(lines[0]) if lines else ""

    if first_line in LABELS:
        return first_line

    if first_line.startswith("label:"):
        value = first_line.split(":", 1)[1].strip().lower()
        value = value.replace("non type error", "non-type error")

        if value in LABELS:
            return value

    # 优先匹配 non-type error，避免被 type error 子串误伤
    if re.search(r"\bnon[- ]type error\b", s):
        return "non-type error"

    if re.search(r"\btype error\b", s):
        return "type error"

    return "INVALID"


def extract_reason(text: str):
    if not text or not text.strip():
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines:
        if line.lower().startswith("reason:"):
            return line.split(":", 1)[1].strip()

    remain = []

    for line in lines:
        low = line.lower()
        if low.startswith("label:"):
            continue
        if low in LABELS:
            continue
        remain.append(line)

    return " ".join(remain).strip()


# =========================
# 6. GPT API 调用
# =========================

def build_headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def get_token_param_name() -> str:
    return TOKEN_PARAMETER


def call_gpt(messages: list[dict], max_retries=MAX_RETRIES) -> tuple[str, dict]:
    import requests
    for attempt in range(1, max_retries + 1):
        try:
            token_param = get_token_param_name()

            payload = {
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": API_TEMPERATURE,
                token_param: MAX_OUTPUT_TOKENS
            }

            if API_TOP_P is not None:
                payload["top_p"] = API_TOP_P
            resp = requests.post(
                BASE_URL,
                headers=build_headers(),
                json=payload,
                timeout=(10, REQUEST_TIMEOUT),
            )

            if resp.status_code == 200:
                data = resp.json()

                usage = data.get("usage", {})
                if PRINT_USAGE and usage:
                    print("usage:", usage)

                    prompt_details = usage.get("prompt_tokens_details", {})
                    cached_tokens = prompt_details.get("cached_tokens")
                    if cached_tokens is not None:
                        print("cached_tokens:", cached_tokens)

                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )

                if content:
                    return content.strip(), usage

                print(f"返回为空，第 {attempt} 次重试")
                print(str(data)[:1000])

            else:
                print(f"API 请求失败，第 {attempt} 次重试: {resp.status_code}")
                print(resp.text[:1500])

                # 如果中转不支持 max_completion_tokens，可以自动退回 max_tokens
                if "max_completion_tokens" in resp.text and "unsupported" in resp.text.lower():
                    print("提示：接口可能不支持 max_completion_tokens，请把 get_token_param_name() 改成 'max_tokens'。")

        except Exception as e:
            print(f"请求异常，第 {attempt} 次重试: {e}")

        time.sleep(SLEEP_SECONDS * attempt)

    raise RuntimeError("GPT API 多次请求失败，已终止。")


def reprompt_for_label(raw_output: str, max_retries=2) -> tuple[str, dict]:
    messages = [
        {
            "role": "system",
            "content": "You are an expert in code translation error analysis. Your task is to extract the final label only.",
        },
        {
            "role": "user",
            "content": (
                "Based on the following analysis, output the final classification again.\n"
                "You must follow this exact format:\n"
                "Label: type error\n"
                "or\n"
                "Label: non-type error\n"
                "Reason: <brief explanation>\n"
                "Do not output any text before the Label line.\n\n"
                f"Previous analysis:\n{raw_output}"
            ),
        },
    ]

    return call_gpt(messages, max_retries=max_retries)


# =========================
# 7. 评估函数
# =========================

def evaluate_results(results: list[dict]):
    from sklearn.metrics import accuracy_score, recall_score, classification_report
    valid_results = [
        r for r in results
        if r.get("gold") in LABELS and r.get("pred") in LABELS
    ]

    invalid_count = len(results) - len(valid_results)

    if not valid_results:
        print("\n没有可评测的有效样本，请检查 gold 标签和模型输出。")
        return

    golds = [r["gold"] for r in valid_results]
    preds = [r["pred"] for r in valid_results]

    print("\n===== Evaluation =====")
    print("Mode:", MODE)
    print("Model:", MODEL_NAME)
    print("Valid samples:", len(valid_results))
    print("Invalid outputs:", invalid_count)
    print("Accuracy:", accuracy_score(golds, preds))
    print("Binary Recall:", recall_score(golds, preds, pos_label="type error"))
    print(classification_report(golds, preds, digits=4, zero_division=0))


# =========================
# 8. 主流程
# =========================

def main():
    configure_identification()
    validate_mode(MODE)
    check_few_shot_examples()

    print(f"当前模式: {MODE}")
    print(f"模型: {MODEL_NAME}")
    print(f"输出 reason: {OUTPUT_REASON}")
    print("GPT prompt caching: 自动生效；看 usage.prompt_tokens_details.cached_tokens")
    print(f"输入文件: {TEST_FILE}")
    print(f"输出文件: {OUTPUT_FILE}")

    test_data = load_jsonl(TEST_FILE)
    if SAMPLE_LIMIT is not None:
        test_data = test_data[:SAMPLE_LIMIT]
    print(f"读取测试样本 {len(test_data)} 条")

    static_user_prefix = build_static_user_prefix(MODE)

    existing_results = {}
    if RESUME_FROM_OUTPUT:
        existing_results = load_existing_results(OUTPUT_FILE)
        print(f"检测到已有结果 {len(existing_results)} 条，将跳过这些 id。")

    current_results = []

    for idx, sample in enumerate(test_data, start=1):
        sample_key = get_sample_key(sample)

        if RESUME_FROM_OUTPUT and sample_key in existing_results:
            old = existing_results[sample_key]
            current_results.append(old)
            print(f"[{idx}/{len(test_data)}] id={sample.get('id', '')} 已存在，跳过")
            continue

        messages = build_messages(static_user_prefix, sample, MODE)

        try:
            generated, usage = call_gpt(messages)
        except Exception as e:
            print(f"[{idx}/{len(test_data)}] id={sample.get('id', '')} API 失败: {e}")
            print("该样本未写入结果文件，下次运行会自动重试。")
            continue

        pred = extract_label(generated)

        if pred == "INVALID":
            print("\n===== INVALID RAW OUTPUT BEFORE REPAIR =====")
            print(generated)
            print("===== END =====\n")

            try:
                repaired, repair_usage = reprompt_for_label(generated)
                repaired_pred = extract_label(repaired)

                if repaired_pred != "INVALID":
                    generated = repaired
                    pred = repaired_pred
                else:
                    print("修复后仍然 INVALID，修复输出如下：")
                    print(repaired)

            except Exception as e:
                print(f"修复 label 时失败: {e}")

        reason = extract_reason(generated) if OUTPUT_REASON else ""

        result = {
            "mode": MODE,
            "model": MODEL_NAME,
            "id": sample.get("id", ""),
            "source_file": sample.get("source_file", ""),
            "target_file": sample.get("target_file", ""),
            "gold": get_gold_label(sample),
            "pred": pred,
            "reason": reason,
            "raw_output": generated,
        }

        current_results.append(result)

        if SAVE_EVERY_RESULT:
            append_jsonl(result, OUTPUT_FILE)

        print(
            f"[{idx}/{len(test_data)}] "
            f"id={result['id']} gold={result['gold']} pred={pred}"
        )

    if not SAVE_EVERY_RESULT:
        save_jsonl(current_results, OUTPUT_FILE)
        print(f"\n预测结果已保存到: {OUTPUT_FILE}")
    else:
        print(f"\n预测结果已逐条保存到: {OUTPUT_FILE}")

    all_results = load_jsonl(OUTPUT_FILE) if OUTPUT_FILE.exists() else current_results
    evaluate_results(all_results)



SAMPLE_LIMIT = None
RESUME_FROM_OUTPUT = True
API_TEMPERATURE = 0
API_TOP_P = 1
MAX_OUTPUT_TOKENS = 128
TOKEN_PARAMETER = 'max_completion_tokens'

def configure_identification():
    import argparse
    import hashlib
    global API_KEY, BASE_URL, MODEL_NAME, MODE, OUTPUT_REASON, TEST_FILE, OUTPUT_FILE, JOBS
    global REQUEST_TIMEOUT, MAX_RETRIES, SLEEP_SECONDS, PRINT_USAGE
    global API_TEMPERATURE, API_TOP_P, MAX_OUTPUT_TOKENS, TOKEN_PARAMETER, RESUME_FROM_OUTPUT, SAMPLE_LIMIT
    parser = ConfigArgumentParser(description='Type-error identification with GPT; original prompt text and examples are preserved.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', default=MODEL_NAME)
    parser.add_argument('--mode', choices=['zero_shot', 'define_only', 'few_shot_only', 'define_and_few_shot', 'define_and_few_shot_and_error_message'], default=MODE)
    parser.add_argument('--output-reason', action=argparse.BooleanOptionalAction, default=OUTPUT_REASON)
    parser.add_argument('--base-url', default=__import__('os').environ.get('GPT_BASE_URL', BASE_URL), help='Full API endpoint; Gemini can use a {model} placeholder.')
    parser.add_argument('--api-key-env', default='GPT_API_KEY', help='Environment variable NAME; never put a key in a config file.')
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top-p', type=float, default=None)
    parser.add_argument('--max-output-tokens', type=int, default=None)
    parser.add_argument('--token-parameter', choices=['max_completion_tokens', 'max_tokens'], default='max_completion_tokens')
    parser.add_argument('--request-timeout', type=int, default=REQUEST_TIMEOUT)
    parser.add_argument('--max-retries', type=int, default=MAX_RETRIES)
    parser.add_argument('--sleep-seconds', type=float, default=SLEEP_SECONDS)
    parser.add_argument('--print-usage', action=argparse.BooleanOptionalAction, default=PRINT_USAGE)
    parser.add_argument('--prompt-cache', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    MODEL_NAME, MODE, OUTPUT_REASON = args.model, args.mode, args.output_reason
    RESUME_FROM_OUTPUT, SAMPLE_LIMIT = args.resume, args.limit
    if args.limit is not None and args.limit < 0:
        parser.error('--limit must be nonnegative')
    if args.out.exists() and not args.resume and not args.dry_run:
        raise ValueError('Output exists; use --resume or a new output path')
    TEST_FILE, OUTPUT_FILE = args.input, args.out
    JOBS = [{'name': args.input.stem, 'test_file': args.input, 'output_file': args.out}]
    BASE_URL = args.base_url.replace('{model}', MODEL_NAME)
    REQUEST_TIMEOUT, MAX_RETRIES, SLEEP_SECONDS = args.request_timeout, args.max_retries, args.sleep_seconds
    PRINT_USAGE = args.print_usage
    API_TEMPERATURE, API_TOP_P = args.temperature, args.top_p
    MAX_OUTPUT_TOKENS = args.max_output_tokens if args.max_output_tokens is not None else (256 if OUTPUT_REASON else 128)
    TOKEN_PARAMETER = args.token_parameter
    globals()['USE_PROMPT_CACHE'] = args.prompt_cache
    if MAX_OUTPUT_TOKENS <= 0:
        parser.error('--max-output-tokens must be positive')
    args.max_output_tokens = MAX_OUTPUT_TOKENS
    active = get_active_few_shot_examples(MODE)
    extra = {'prompt_sha256': hashlib.sha256(get_system_prompt(MODE).encode()).hexdigest(),
             'few_shot_sha256': hashlib.sha256(json.dumps(active, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
             'few_shot_count': len(active), 'output_instruction': get_output_instruction()}
    finish_setup(args, args.out.with_suffix('.run.json'), extra)
    API_KEY = credential(args.api_key_env)


if __name__ == "__main__":
    main()
