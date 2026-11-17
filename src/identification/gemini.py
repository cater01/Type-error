import json
import time
import re
from pathlib import Path


# =========================
# 1. 基本配置
# =========================


# Shared CLI/config support; works when invoked from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from runtime import (REPO_ROOT, ConfigArgumentParser, generation_datasets,
                     external_root, configure_java, finish_setup, seed_torch,
                     credential, repo_path)

API_KEY = ''

MODEL_NAME = "gemini-3.1-pro-preview"

# Gemini native API: streamGenerateContent
BASE_URL = 'https://yunwu.ai/v1beta/models/{model}:streamGenerateContent?alt=sse'

WORKSPACE = REPO_ROOT

JOBS = []

# 可选模式：
# "zero_shot"
# "define_only"
# "few_shot_only"
# "define_and_few_shot"
# "define_and_few_shot_and_error_message"
MODE = "define_and_few_shot_and_error_message"

# 大批量第一轮建议 False；分歧样本/抽样复查可以 True
OUTPUT_REASON = True

REQUEST_TIMEOUT = 60
MAX_RETRIES = 4
SLEEP_SECONDS = 2

# 是否打印 Gemini usage 信息
PRINT_USAGE = False


# =========================
# 2. 标签集合
# =========================

LABELS = [
    "type error",
    "non-type error",
]


# =========================
# 3. few-shot 示例
# =========================

FEW_SHOT_EXAMPLES = [
    {
        "id": "3",
        "source_file": "00e90909a77ce9e22bb7cbf1285b0609_2.py",
        "target_file": "00e90909a77ce9e22bb7cbf1285b0609_2.java",
        "source_code": "\n\nk,p=map(int,input().split())\ns=0\nfor i in range(1,k+1):\n    s+=int(str(i)+str(i)[::-1])\nprint(s%p)\n",
        "translated_code": "import java.util.*;\n\npublic class Main {\n    public static void main(String[] args) {\n        Scanner scanner = new Scanner(System.in);\n        int k = scanner.nextInt();\n        int p = scanner.nextInt();\n        int s = 0;\n        for (int i = 1; i <= k; i++) {\n            s += Integer.parseInt(String.valueOf(i) + new StringBuilder().append(i).reverse().toString());\n        }\n        System.out.println(s % p);\n    }\n}",
        "label_0": "type error",
        "note": "Type error: attempted to convert a string to an int, but the value was too large and exceeded the allowable range.",
        "Error message": "RUNTIME_ERROR: Exception in thread \"main\" java.lang.NumberFormatException: For input string: \"2147557412\"\r\n\tat java.base/java.lang.NumberFormatException.forInputString(NumberFormatException.java:67)\r\n\tat java.base/j",
    },
    {
        "id": "60",
        "source_file": "COMPUTE_AVERAGE_TWO_NUMBERS_WITHOUT_OVERFLOW_1.py",
        "target_file": "COMPUTE_AVERAGE_TWO_NUMBERS_WITHOUT_OVERFLOW_1.java",
        "source_code": "def compute_average ( a , b ) :\n    return ( a // 2 ) + ( b // 2 ) + ( ( a % 2 + b % 2 ) // 2 )",
        "translated_code": "public class Main {\n    public static void main(String[] args) {\n        System.out.println(computeAverage(5, 7));\n    }\n\n    public static double computeAverage(int a, int b) {\n        return (a / 2.0) + (b / 2.0) + ((a % 2 + b % 2) / 2.0);\n    }\n}",
        "label_0": "type error",
        "note": "Type error: The return type of the translated Java function is double, while the expected behavior corresponds to an integer result. The return types are incompatible.",
        "Error message": "OUTPUT_MISMATCH: #Results:1, 10",
    },
    {
        "id": "135",
        "source_file": "s043545989.py",
        "target_file": "s043545989.java",
        "source_code": "import collections\nN = int(input())\na = list(map(int,input().split()))\nB = collections.Counter(a)\nk = list(B.keys())\nv = list(B.values())\n\nif len(B) == 3 and k[0]^k[1]^k[2] == 0 and v[0] == v[1] == v[2]:\n    print(\"Yes\")\nelif len(B) == 2 and 0 in k and (v[0] == 2*v[1] or v[1] == 2*v[0]):\n    print(\"Yes\")\nelif len(B) == 1 and 0 in k:\n    print(\"Yes\")\nelse: print(\"No\")",
        "translated_code": "import java.util.*;\n\npublic class Main {\n    public static void main(String[] args) {\n        Scanner sc = new Scanner(System.in);\n        int N = sc.nextInt();\n        int[] a = new int[N];\n        for (int i = 0; i < N; i++) {\n            a[i] = sc.nextInt();\n        }\n\n        Map<Integer, Integer> B = new HashMap<>();\n        for (int i : a) {\n            if (B.containsKey(i)) {\n                B.put(i, B.get(i) + 1);\n            } else {\n                B.put(i, 1);\n            }\n        }\n\n        List<Integer> k = new ArrayList<>(B.keySet());\n        List<Integer> v = new ArrayList<>(B.values());\n\n        if (B.size() == 3 && k.get(0)^k.get(1)^k.get(2) == 0 && v.get(0).equals(v.get(1)) && v.get(1).equals(v.get(2))) {\n            System.out.println(\"Yes\");\n        } else if (B.size() == 2 && k.contains(0) && (v.get(0) == 2*v.get(1) || v.get(1) == 2*v.get(0))) {\n            System.out.println(\"Yes\");\n        } else if (B.size() == 1 && k.contains(0)) {\n            System.out.println(\"Yes\");\n        } else {\n            System.out.println(\"No\");\n        }\n    }\n}",
        "label_0": "type error",
        "note": "Type error: The binary operator ^ is applied in a way that causes an int and a boolean expression to be combined incorrectly.",
        "Error message": "COMPILE_ERROR: Main.java:24: 错误: 二元运算符 '^' 的操作数类型错误\n        if (B.size() == 3 && k.get(0)^k.get(1)^k.get(2) == 0 && v.get(0).equals(v.get(1)) && v.get(1).equals(v.get(2))) {\n                                              ^\n  第一个类型:  int\n  第二个类型: boolean\n1 个错误",
    },
    {
        "id": "18",
        "source_file": "0ddc839e17dee20e1a954c1289de7fbd_2.py",
        "target_file": "0ddc839e17dee20e1a954c1289de7fbd_2.java",
        "source_code": "g, v = [input() for i in range(8)], [8, 8]\nfor i in range(8):\n    for j in range(8):\n        if g[i][j] == 'W' and all(g[k][j] == '.' for k in range(i)):\n            v[0] = min(v[0], i)\n        elif g[i][j] == 'B' and all(g[k][j] == '.' for k in range(i + 1, 8)):\n            v[1] = min(v[1], 7 - i)\nprint('A' if v[0] <= v[1] else 'B')",
        "translated_code": "import java.util.Scanner;\n\npublic class Main {\n    public static void main(String[] args) {\n        Scanner scanner = new Scanner(System.in);\n        char[][] board = new char[8][8];\n        int[] whiteRow = {8, 8};\n        int[] blackRow = {8, 8};\n\n        for (int i = 0; i < 8; i++) {\n            String line = scanner.nextLine();\n            for (int j = 0; j < 8; j++) {\n                board[i][j] = line.charAt(j);\n            }\n        }\n\n        for (int i = 0; i < 8; i++) {\n            for (int j = 0; j < 8; j++) {\n                if (board[i][j] == 'W' && isClearPath(board, i, j)) {\n                    whiteRow[0] = Math.min(whiteRow[0], i);\n                } else if (board[i][j] == 'B' && isClearPath(board, i, j)) {\n                    blackRow[0] = Math.min(blackRow[0], 7 - i);\n                }\n            }\n        }\n\n        System.out.println(whiteRow[0] <= blackRow[0] ? \"A\" : \"B\");\n    }\n\n    private static boolean isClearPath(char[][] board, int i, int j) {\n        for (int k = 0; k < i; k++) {\n            if (board[k][j] != '.') {\n                return false;\n            }\n        }\n        return true;\n    }\n}",
        "label_0": "non-type error",
        "note": "Non-type error: The logic for checking the black piece path is incorrect.",
        "Error message": "OUTPUT_MISMATCH: exp_any='B', got='A'",
    },
    {
        "id": "20",
        "source_file": "0efe9afd8e6be9e00f7949be93f0ca1a_2.py",
        "target_file": "0efe9afd8e6be9e00f7949be93f0ca1a_2.java",
        "source_code": "n = int(input())\nboard = sorted(map(int,input().split()))\nb = w = 0\nfor i in range(n//2):\n    b += abs(board[-i-1] - (n-i*2))\n    w += abs(board[-i-1] - (n-1-i*2))\nprint(min(b,w))",
        "translated_code": "import java.util.*;\n\npublic class Main {\n    public static void main(String[] args) {\n        Scanner in = new Scanner(System.in);\n        int n = in.nextInt();\n        Integer[] board = new Integer[n];\n        for(int i = 0; i < n; i++) {\n            board[i] = in.nextInt();\n        }\n        Arrays.sort(board, Collections.reverseOrder());\n        int b = 0;\n        int w = 0;\n        for(int i = 0; i < n/2; i++) {\n            b += Math.abs(board[i] - (n-i*2));\n            w += Math.abs(board[i] - (n-1-i*2));\n        }\n        System.out.println(Math.min(b, w));\n    }\n}",
        "label_0": "non-type error",
        "note": "Non-type error: The program attempts to read more input than is actually available.",
        "Error message": "RUNTIME_ERROR: Exception in thread \"main\" java.util.NoSuchElementException\r\n\tat java.base/java.util.Scanner.throwFor(Scanner.java:937)\r\n\tat java.base/java.util.Scanner.next(Scanner.java:1594)\r\n\tat java.base/java.util.Scanner.nextInt(Scanner.java:2258)",
    },
]


# =========================
# 4. Prompt 配置
# =========================

ZERO_SHOT_SYSTEM_PROMPT = """
You are an expert in code translation error analysis.

Your task is to classify whether the Java translation error is a type error or a non-type error.

You must follow the user's required output format strictly.
""".strip()


DEFINE_SYSTEM_PROMPT = """
You are an expert in code translation error analysis.

Your task is to classify whether the Java translation error is a type error or a non-type error.

Definitions:

- type error:
Type error refers to cases where the code is not correctly translated due to the use of an inappropriate or incorrect data type.
Using a more suitable or correct data type would avoid the error.

It includes:
1. The translated code uses a data type whose precision or numeric range is insufficient, leading to incorrect results during execution.
2. The types of variables, function parameters, or return values in the translated code are inconsistent with the declared or expected types in the target language.
3. The translated code performs an unsupported or incompatible operation on a value of a certain type.

- non-type error:
The main problem is not caused by the use of an inappropriate data type, but by other translation issues such as logic, algorithm, control flow, condition checking, indexing, data handling, input handling, or output-related mistakes.

Special rule:
If the Java error message indicates that a numeric string cannot be parsed by Integer.parseInt, Long.parseLong, or a similar function because the value is too large for the target numeric type, classify it as a type error rather than a non-type error.

Important:
Use the error message as additional evidence, but classify based on the root cause of the translation error.
Do not classify based only on surface keywords.

You must follow the user's required output format strictly.
""".strip()

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


def get_output_instruction() -> str:
    if OUTPUT_REASON:
        return (
            "Output format requirements:\n"
            "The first line must be exactly one of:\n"
            "Label: type error\n"
            "Label: non-type error\n"
            "The second line must start with:\n"
            "Reason: <用中文说明分类依据，包含：1. 触发错误的具体代码位置或变量；2. 错误是否由数据类型、精度、范围、返回值/参数类型或类型不兼容导致；3. 为什么不是另一类错误。控制在3-5句话内。>\n"
            "The reason must be written in Chinese, while the label must remain exactly in English.\n"
            "Do not output markdown, bullets, code blocks, JSON, or any text before the Label line."
        )

    return (
        "Output format requirements:\n"
        "Return exactly one line and nothing else:\n"
        "type error\n"
        "or\n"
        "non-type error\n"
        "Do not output markdown, bullets, code blocks, JSON, explanations, or any extra text."
    )


def build_user_prompt(sample: dict, mode: str) -> str:
    active_examples = get_active_few_shot_examples(mode)
    use_error_message = mode == "define_and_few_shot_and_error_message"

    parts = []

    parts.append(
        "Given the following Python source code, translated Java code, "
        "and possibly a compiler/runtime/output-mismatch message, classify whether "
        "the Java translation error is a type error or a non-type error."
    )
    parts.append("")

    if use_error_message:
        parts.append(
            "You may use the error message as additional evidence, but classify based on "
            "the root cause of the translation error rather than surface keywords only."
        )
        parts.append("")

    parts.append(get_output_instruction())
    parts.append("")

    if active_examples:
        parts.append("Examples:")
        parts.append("")

        for i, ex in enumerate(active_examples, start=1):
            parts.append(f"Example {i}:")
            parts.append("Python:")
            parts.append(ex.get("source_code", ""))
            parts.append("")

            parts.append("Java:")
            parts.append(ex.get("translated_code", ""))
            parts.append("")

            if use_error_message:
                parts.append("Error message:")
                parts.append(ex.get("Error message", ""))
                parts.append("")

            if OUTPUT_REASON:
                parts.append(f"Label: {ex['label_0']}")
                if ex.get("note", "").strip():
                    parts.append(f"Reason: {ex['note'].strip()}")
            else:
                parts.append(ex["label_0"])

            parts.append("")

        parts.append("Now classify the following sample:")
    else:
        parts.append("Classify the following sample:")

    parts.append("")
    parts.append("Python:")
    parts.append(sample.get("source_code", ""))
    parts.append("")

    parts.append("Java:")
    parts.append(sample.get("translated_code", ""))
    parts.append("")

    if use_error_message:
        parts.append("Error message:")
        parts.append(sample.get("Error message", ""))
        parts.append("")

    parts.append("Final answer:")
    parts.append(get_output_instruction())

    return "\n".join(parts)


# =========================
# 5. 文件工具函数
# =========================

def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} 第 {line_num} 行 JSON 解析失败: {e}")

    return rows


def append_jsonl(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_results(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"警告：已有结果文件第 {line_num} 行不是合法 JSON，已跳过。")

    return rows


# =========================
# 6. 校验函数
# =========================

def validate_mode(mode: str) -> None:
    allowed = {
        "zero_shot",
        "define_only",
        "few_shot_only",
        "define_and_few_shot",
        "define_and_few_shot_and_error_message",
    }

    if mode not in allowed:
        raise ValueError(f"MODE 必须是 {allowed} 之一，当前是: {mode}")


def check_few_shot_examples() -> None:
    for ex in FEW_SHOT_EXAMPLES:
        if ex.get("label_0") not in LABELS:
            raise ValueError(
                f"few-shot 示例 {ex.get('id')} 的 label_0 不合法: {ex.get('label_0')}"
            )


# =========================
# 7. 标签解析
# =========================

def normalize_text(text: str) -> str:
    s = text.strip().lower()

    s = s.replace("\u2011", "-")
    s = s.replace("\u2012", "-")
    s = s.replace("\u2013", "-")
    s = s.replace("\u2014", "-")

    s = re.sub(r"[*_`#>\[\]\{\}\(\)]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def extract_label(text: str) -> str:
    if not text or not text.strip():
        return "INVALID"

    s = normalize_text(text)

    # 0. 处理 Gemini 截断输出
    # 因为本任务只有两个标签：
    # type error / non-type error
    # 所以 non / non- / non-type 可以安全视为 non-type error
    if s in {"non", "non-", "non-type", "non type", "non type error"}:
        return "non-type error"

    if s in {"type", "type-error"}:
        return "type error"

    # 1. 优先识别 Label: xxx
    m = re.search(
        r'\blabel\s*["\']?\s*[:：=]\s*["\']?\s*(non[- ]?type error|type error)\b',
        s
    )
    if m:
        value = m.group(1).replace("non type error", "non-type error")
        return value

    # 2. 识别 prediction / answer / classification / class
    m = re.search(
        r'\b(prediction|answer|classification|class)\s*["\']?\s*[:：=]\s*["\']?\s*(non[- ]?type error|type error)\b',
        s
    )
    if m:
        value = m.group(2).replace("non type error", "non-type error")
        return value

    # 3. 纯标签输出
    if re.fullmatch(r"non[- ]?type error", s):
        return "non-type error"

    if re.fullmatch(r"type error", s):
        return "type error"

    # 4. 常见否定表达
    if re.search(r"\b(not a type error|not type error|no type error)\b", s):
        return "non-type error"

    # 5. 宽松识别，non-type 必须放在 type 前面
    if re.search(r"\bnon[- ]?type error\b", s):
        return "non-type error"

    if re.search(r"\btype error\b", s):
        return "type error"

    return "INVALID"


def extract_reason(text: str) -> str:
    if not text or not text.strip():
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines:
        low = line.lower().strip()
        if low.startswith("reason:"):
            reason = line.split(":", 1)[1].strip()
            return "" if reason.lower() == "reason" else reason
        if low.startswith("reason："):
            reason = line.split("：", 1)[1].strip()
            return "" if reason.lower() == "reason" else reason

    remain = []

    for line in lines:
        low = line.lower().strip()

        if low.startswith("label:"):
            continue

        if low in {"reason", "reason:", "reason："}:
            continue

        if low in LABELS:
            continue

        remain.append(line)

    return " ".join(remain).strip()


# =========================
# 8. Gemini Native API 调用
# =========================

def build_gemini_payload(system_prompt: str, user_prompt: str) -> dict:
    """
    Gemini native request body.
    不改变原来的分类功能，只把 OpenAI messages 格式换成 Gemini contents 格式。
    """
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": user_prompt
                    }
                ]
            }
        ],
        "systemInstruction": {
            "role": "user",
            "parts": [
                {
                    "text": system_prompt
                }
            ]
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "OFF"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "OFF"
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "OFF"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "OFF"
            },
            {
                "category": "HARM_CATEGORY_CIVIC_INTEGRITY",
                "threshold": "BLOCK_NONE"
            }
        ],
        "tools": [],
        "generationConfig": {
            # 分类任务建议低温度，更稳定
            "temperature": API_TEMPERATURE,
            "topP": API_TOP_P,
            "maxOutputTokens": MAX_OUTPUT_TOKENS,

            # 这里不要 includeThoughts，否则可能返回额外思考内容，增加成本和解析复杂度
            #"thinkingConfig": {
            #    "includeThoughts": False,
            #    "thinkingBudget": 1024
            #}
        }
    }


def parse_gemini_sse_response(text: str) -> tuple[str, dict]:
    """
    解析 Gemini streamGenerateContent 的 SSE 返回。

    常见格式：
    data: {"candidates":[{"content":{"parts":[{"text":"..."}]}}]}

    返回：
    - 合并后的文本
    - 最后一次出现的 usageMetadata
    """
    output_parts = []
    usage_metadata = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("data:"):
            line = line[len("data:"):].strip()

        if not line or line == "[DONE]":
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if "usageMetadata" in obj:
            usage_metadata = obj.get("usageMetadata", {}) or usage_metadata

        candidates = obj.get("candidates", [])
        if not candidates:
            continue

        for cand in candidates:
            content = cand.get("content", {})
            parts = content.get("parts", [])

            for part in parts:
                # 不读取 thought，只读取正式 text
                if part.get("thought") is True:
                    continue

                text_part = part.get("text", "")
                if text_part:
                    output_parts.append(text_part)

    return "".join(output_parts).strip(), usage_metadata


def call_api(system_prompt: str, user_prompt: str, max_retries: int = MAX_RETRIES) -> str:
    import requests
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            payload = build_gemini_payload(system_prompt, user_prompt)

            resp = requests.post(
                BASE_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            if resp.status_code == 200:
                response_text = resp.content.decode("utf-8", errors="replace")
                content, usage = parse_gemini_sse_response(response_text)

                if PRINT_USAGE and usage:
                    print("usage:", usage)

                if content and content.strip():
                    return content.strip()

                print(f"返回为空，第 {attempt} 次重试")
                print(resp.text[:1000])

            else:
                print(f"API 请求失败，第 {attempt} 次重试: {resp.status_code}")
                print(resp.text[:1000])

        except Exception as e:
            last_error = e
            print(f"请求异常，第 {attempt} 次重试: {e}")

        time.sleep(SLEEP_SECONDS * attempt)

    raise RuntimeError(f"Gemini API 多次请求失败，已终止。最后错误: {last_error}")


def reprompt_for_label(raw_output: str, max_retries: int = 2) -> str:
    repair_system_prompt = "You extract classification labels only. Do not explain."

    repair_user_prompt = (
        "Extract the final classification label from the following answer.\n"
        "Return exactly one line and nothing else:\n"
        "type error\n"
        "or\n"
        "non-type error\n\n"
        f"Answer:\n{raw_output}"
    )

    return call_api(
        system_prompt=repair_system_prompt,
        user_prompt=repair_user_prompt,
        max_retries=max_retries,
    )


# =========================
# 9. 预测统计
# =========================

def print_prediction_summary(results: list[dict]) -> None:
    total = len(results)

    valid = sum(1 for r in results if r.get("pred") in LABELS)
    invalid = total - valid

    type_count = sum(1 for r in results if r.get("pred") == "type error")
    non_type_count = sum(1 for r in results if r.get("pred") == "non-type error")

    print("\n===== Prediction Summary =====")
    print("Mode:", MODE)
    print("Model:", MODEL_NAME)
    print("Total samples:", total)
    print("Valid predictions:", valid)
    print("Invalid predictions:", invalid)
    print("Predicted type error:", type_count)
    print("Predicted non-type error:", non_type_count)

    if valid > 0:
        print("Type error ratio:", type_count / valid)
        print("Non-type error ratio:", non_type_count / valid)


# =========================
# 10. 主流程
# =========================

def run_job(job: dict) -> None:
    test_file = job["test_file"]
    output_file = job["output_file"]

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n===== Job: {job['name']} =====")
    print(f"输入文件: {test_file}")
    print(f"输出文件: {output_file}")

    test_data = load_jsonl(test_file)
    if SAMPLE_LIMIT is not None:
        test_data = test_data[:SAMPLE_LIMIT]
    print(f"读取样本 {len(test_data)} 条")

    # 断点续跑：每个数据集写入独立输出文件，因此相同 id 不会跨 Qwen7B/Qwen32B 互相跳过。
    existing_results = load_existing_results(output_file) if RESUME_FROM_OUTPUT else []
    done_ids = {str(r.get("id", "")) for r in existing_results if str(r.get("id", ""))}

    if existing_results:
        print(f"检测到已有结果 {len(existing_results)} 条，将跳过这些 id。")
        results = existing_results[:]
    else:
        results = []

    system_prompt = get_system_prompt(MODE)

    for idx, sample in enumerate(test_data, start=1):
        sample_id = str(sample.get("id", ""))

        # 如果原数据没有 id，就用行号当 id
        if not sample_id:
            sample_id = str(idx)

        if sample_id in done_ids:
            print(f"[{idx}/{len(test_data)}] id={sample_id} 已存在，跳过")
            continue

        user_prompt = build_user_prompt(sample, MODE)

        try:
            generated = call_api(system_prompt, user_prompt)
        except Exception as e:
            print(f"[{idx}/{len(test_data)}] id={sample_id} API 失败: {e}")

            result = {
                "mode": MODE,
                "model": MODEL_NAME,
                "id": sample_id,
                "source_file": sample.get("source_file", ""),
                "target_file": sample.get("target_file", ""),
                "pred": "INVALID",
                "reason": "",
                "raw_output": "",
                "error": str(e),
            }

            results.append(result)
            append_jsonl(result, output_file)
            continue

        pred = extract_label(generated)

        # 第一次解析失败，再让模型从自己的输出中提取标签
        if pred == "INVALID":
            print("\n===== INVALID RAW OUTPUT BEFORE REPAIR =====")
            print(generated)
            print("===== END =====\n")

            try:
                repaired = reprompt_for_label(generated)
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
            "id": sample_id,
            "source_file": sample.get("source_file", ""),
            "target_file": sample.get("target_file", ""),
            "gold": sample.get("gold", sample.get("label_0", "")),
            "pred": pred,
            "reason": reason,
            "raw_output": generated,
        }

        results.append(result)
        append_jsonl(result, output_file)

        print(
            f"[{idx}/{len(test_data)}] "
            f"id={sample_id} pred={pred}"
        )

    print(f"\n预测结果已保存到: {output_file}")
    print_prediction_summary(results)


def main() -> None:
    configure_identification()
    validate_mode(MODE)
    check_few_shot_examples()

    print(f"当前模式: {MODE}")
    print(f"模型: {MODEL_NAME}")
    print(f"输出 reason: {OUTPUT_REASON}")
    print("Gemini API: configured endpoint")

    for job in JOBS:
        run_job(job)



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
    parser = ConfigArgumentParser(description='Type-error identification with GEMINI; original prompt text and examples are preserved.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', default=MODEL_NAME)
    parser.add_argument('--mode', choices=['zero_shot', 'define_only', 'few_shot_only', 'define_and_few_shot', 'define_and_few_shot_and_error_message'], default=MODE)
    parser.add_argument('--output-reason', action=argparse.BooleanOptionalAction, default=OUTPUT_REASON)
    parser.add_argument('--base-url', default=__import__('os').environ.get('GEMINI_BASE_URL', BASE_URL), help='Full API endpoint; Gemini can use a {model} placeholder.')
    parser.add_argument('--api-key-env', default='GEMINI_API_KEY', help='Environment variable NAME; never put a key in a config file.')
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top-p', type=float, default=1)
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
    MAX_OUTPUT_TOKENS = args.max_output_tokens if args.max_output_tokens is not None else (1024 if OUTPUT_REASON else 128)
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
