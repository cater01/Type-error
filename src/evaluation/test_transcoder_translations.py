import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path



# Shared CLI/config support; works when invoked from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from runtime import (REPO_ROOT, ConfigArgumentParser, generation_datasets,
                     external_root, configure_java, finish_setup, seed_torch,
                     credential, repo_path)

WORKSPACE = REPO_ROOT

# =========================
# Path Configuration
# =========================
# Fill these paths directly when running on AutoDL or another machine.
# Leave a value as None to use the default path below it.
FILLED_JAVA_DIR = ''
# Example:

TEST_RESULTS_JSONL = ''
# Example:


DEFAULT_FILLED_DIR = REPO_ROOT / 'results/generation/transcoder/filled_java'
DEFAULT_OUT = REPO_ROOT / 'results/evaluation/transcoder/test_results.jsonl'

JDK_BIN_DIR = None
JAVA_CMD = 'java'
JAVAC_CMD = 'javac'


RESULT_RE = re.compile(r"#Results:\s*(\d+)\s*,\s*(\d+)")


def short_workdir_name(class_name: str) -> str:
    digest = hashlib.sha1(class_name.encode("utf-8")).hexdigest()[:10]
    return f"case_{digest}"


def strip_markdown(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:java)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text


def find_matching_brace(text: str, open_index: int) -> int:
    depth = 0
    in_string = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    escaped = False

    i = open_index
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if in_char:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "'":
                in_char = False
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if ch == "'":
            in_char = True
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1

    raise ValueError("Unbalanced braces")


def parse_f_filled_signature(java_text: str) -> tuple[str, str, list[str], int, int]:
    match = re.search(
        r"((?:public\s+)?static\s+([^;{}\n]+?)\s+f_filled\s*\(([^;{}]*)\))\s*\{",
        java_text,
    )
    if not match:
        raise ValueError("Could not find f_filled signature")
    open_brace = java_text.find("{", match.start())
    close_brace = find_matching_brace(java_text, open_brace)
    signature = " ".join(match.group(1).split())
    return_type = " ".join(match.group(2).split())
    params = [part.strip() for part in match.group(3).split(",") if part.strip()]
    return signature, return_type, params, match.start(), close_brace + 1


def normalize_type(type_text: str) -> str:
    return re.sub(r"\s+", "", type_text).replace("Integer", "int").replace("Boolean", "boolean")


def java_methods(java_text: str) -> list[dict]:
    pattern = re.compile(
        r"(?m)^\s*(?:public\s+|private\s+|protected\s+)?(?:static\s+)?"
        r"([\w<>\[\]]+(?:\s*\[\])?)\s+(\w+)\s*\(([^;{}]*)\)\s*\{"
    )
    methods = []
    for match in pattern.finditer(java_text):
        open_brace = java_text.find("{", match.start())
        close_brace = find_matching_brace(java_text, open_brace)
        methods.append(
            {
                "return_type": " ".join(match.group(1).split()),
                "name": match.group(2),
                "params": [part.strip() for part in match.group(3).split(",") if part.strip()],
                "text": java_text[match.start() : close_brace + 1].strip(),
            }
        )
    return methods


def method_score(method: dict, expected_return_type: str, expected_params: list[str]) -> int:
    if method["name"] == "main":
        return -100
    score = 0
    if method["name"] == "f_filled":
        score += 50
    if normalize_type(method["return_type"]) == normalize_type(expected_return_type):
        score += 20
    if len(method["params"]) == len(expected_params):
        score += 20
    if "System.out.print" in method["text"]:
        score -= 10
    if re.search(r"\bScanner\s*\(", method["text"]):
        score -= 10
    return score


def extract_repair_method(raw_text: str, signature: str, return_type: str, params: list[str]) -> str:
    java_text = strip_markdown(raw_text).replace("@@ Response", "").strip()
    methods = java_methods(java_text)
    if not methods:
        raise ValueError("No Java method found in raw response")
    method = max(methods, key=lambda item: method_score(item, return_type, params))
    if method_score(method, return_type, params) < 0:
        raise ValueError("No suitable Java method found in raw response")

    original_name = method["name"]
    body = method["text"][method["text"].find("{") :].strip()
    if original_name != "f_filled":
        body = re.sub(rf"\b{re.escape(original_name)}\s*\(", "f_filled(", body)
    return f"{signature} {body}"


def repair_from_raw(java_text: str, raw_text: str) -> tuple[str, str]:
    signature, return_type, params, start, end = parse_f_filled_signature(java_text)
    method = extract_repair_method(raw_text, signature, return_type, params)
    return java_text[:start] + method + java_text[end:], "raw_method"


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def remove_old_output(path: Path, keep_old: bool) -> None:
    if path.exists() and not keep_old:
        path.unlink()


def decode_output(value) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value or ""


def kill_process_tree(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def run_command(args: list[str], cwd: Path, timeout: int) -> tuple[str, subprocess.CompletedProcess | None, str]:
    process = None
    try:
        process = subprocess.Popen(
            args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate(timeout=timeout)
        result = subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
        return "ok", result, ""
    except subprocess.TimeoutExpired as exc:
        kill_process_tree(process)
        stdout = decode_output(exc.stdout)
        stderr = decode_output(exc.stderr)
        try:
            extra_stdout, extra_stderr = process.communicate(timeout=2)
            stdout += decode_output(extra_stdout)
            stderr += decode_output(extra_stderr)
        except Exception:
            pass
        return "timeout", None, f"Timeout after {timeout}s\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    except KeyboardInterrupt:
        kill_process_tree(process)
        raise
    except Exception as exc:
        kill_process_tree(process)
        return "launch_error", None, str(exc)


def test_one(java_file: Path, timeout: int, keep_workdirs: bool, work_root: Path, raw_dir: Path | None) -> dict:
    started = time.time()
    class_name = java_file.stem
    record = {
        "name": class_name,
        "java_file": str(java_file),
        "status": "started",
        "compile_returncode": None,
        "run_returncode": None,
        "stdout": "",
        "stderr": "",
        "compile_stdout": "",
        "compile_stderr": "",
        "success_cases": None,
        "total_cases": None,
        "passed": False,
        "repair_status": "not_attempted",
        "repair_error": "",
    }

    try:
        workdir = work_root / short_workdir_name(class_name)
        if workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)
        workdir.mkdir(parents=True, exist_ok=True)

        java_text = java_file.read_text(encoding="utf-8")
        if raw_dir is not None:
            raw_path = raw_dir / f"{class_name}.txt"
            if raw_path.exists():
                try:
                    java_text, repair_status = repair_from_raw(
                        java_text,
                        raw_path.read_text(encoding="utf-8"),
                    )
                    record["repair_status"] = repair_status
                    record["raw_response"] = str(raw_path)
                except Exception as exc:
                    record["repair_status"] = "failed"
                    record["repair_error"] = f"{type(exc).__name__}: {exc}"
            else:
                record["repair_status"] = "missing_raw"

        (workdir / java_file.name).write_text(java_text, encoding="utf-8", newline="\n")
        record["workdir"] = str(workdir)

        compile_status, compile_result, compile_error = run_command(
            [JAVAC_CMD, "-encoding", "UTF-8", "-Xdiags:verbose", java_file.name],
            cwd=workdir,
            timeout=timeout,
        )
        if compile_status != "ok":
            record.update(
                {
                    "status": f"compile_{compile_status}",
                    "compile_stderr": compile_error,
                    "elapsed_seconds": round(time.time() - started, 3),
                }
            )
            return record

        record["compile_returncode"] = compile_result.returncode
        record["compile_stdout"] = compile_result.stdout
        record["compile_stderr"] = compile_result.stderr
        if compile_result.returncode != 0:
            record.update(
                {
                    "status": "compile_error",
                    "elapsed_seconds": round(time.time() - started, 3),
                }
            )
            return record

        run_status, run_result, run_error = run_command(
            [JAVA_CMD, class_name],
            cwd=workdir,
            timeout=timeout,
        )
        if run_status != "ok":
            record.update(
                {
                    "status": f"run_{run_status}",
                    "stderr": run_error,
                    "elapsed_seconds": round(time.time() - started, 3),
                }
            )
            return record

        record["run_returncode"] = run_result.returncode
        record["stdout"] = run_result.stdout
        record["stderr"] = run_result.stderr
        if run_result.returncode != 0:
            record.update(
                {
                    "status": "runtime_error",
                    "elapsed_seconds": round(time.time() - started, 3),
                }
            )
            return record

        match = RESULT_RE.search(run_result.stdout)
        if not match:
            record.update(
                {
                    "status": "missing_results_line",
                    "elapsed_seconds": round(time.time() - started, 3),
                }
            )
            return record

        success = int(match.group(1))
        total = int(match.group(2))
        record["success_cases"] = success
        record["total_cases"] = total
        record["passed"] = success == total and total > 0
        record["status"] = "passed" if record["passed"] else "logic_error"
        record["elapsed_seconds"] = round(time.time() - started, 3)
        return record
    finally:
        if not keep_workdirs:
            shutil.rmtree(workdir, ignore_errors=True)


def summarize(records: list[dict]) -> dict:
    summary = {
        "total_files": len(records),
        "passed_files": 0,
        "compile_errors": 0,
        "runtime_errors": 0,
        "logic_errors": 0,
        "other_errors": 0,
        "success_cases": 0,
        "total_cases": 0,
    }
    for record in records:
        status = record["status"]
        if status == "passed":
            summary["passed_files"] += 1
        elif status.startswith("compile_") or status == "compile_error":
            summary["compile_errors"] += 1
        elif status.startswith("run_") or status == "runtime_error":
            summary["runtime_errors"] += 1
        elif status == "logic_error":
            summary["logic_errors"] += 1
        else:
            summary["other_errors"] += 1

        if isinstance(record.get("success_cases"), int):
            summary["success_cases"] += record["success_cases"]
        if isinstance(record.get("total_cases"), int):
            summary["total_cases"] += record["total_cases"]

    if summary["total_cases"]:
        summary["case_accuracy"] = summary["success_cases"] / summary["total_cases"]
    else:
        summary["case_accuracy"] = None
    return summary


def main() -> int:
    parser = ConfigArgumentParser(description="Compile and run filled TransCoder Java tests.")
    parser.add_argument("--filled-dir", type=Path, default=DEFAULT_FILLED_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory with raw model responses. Defaults to <filled-dir>/../raw_responses.",
    )
    parser.add_argument(
        "--repair-from-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Before compiling, rebuild f_filled from the matching raw response when available.",
    )
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--append", action="store_true", help="Append to existing jsonl instead of replacing it.")
    parser.add_argument(
        "--keep-workdirs",
        action="store_true",
        help="Keep per-file compile directories for debugging.",
    )
    args = parser.parse_args()
    global JAVA_CMD, JAVAC_CMD
    JAVA_CMD, JAVAC_CMD = configure_java(args)
    finish_setup(args, args.out.with_suffix('.run.json'), {'java': JAVA_CMD, 'javac': JAVAC_CMD})

    if args.summary_out is None:
        args.summary_out = args.out.with_suffix(".summary.json")
    if args.raw_dir is None:
        args.raw_dir = args.filled_dir.parent / "raw_responses"
    raw_dir = args.raw_dir if args.repair_from_raw else None

    java_files = sorted(args.filled_dir.glob("*.java"))
    if args.limit is not None:
        java_files = java_files[: args.limit]

    remove_old_output(args.out, keep_old=args.append)
    if args.summary_out.exists() and not args.append:
        args.summary_out.unlink()

    work_root = args.filled_dir.parent / ("test_workdirs" if args.keep_workdirs else "test_tmp")
    work_root.mkdir(parents=True, exist_ok=True)

    print(f"Testing {len(java_files)} Java files from {args.filled_dir}")
    records = []
    for index, java_file in enumerate(java_files, start=1):
        record = test_one(java_file, args.timeout, args.keep_workdirs, work_root, raw_dir)
        records.append(record)
        append_jsonl(args.out, record)
        cases = ""
        if record.get("success_cases") is not None:
            cases = f" ({record['success_cases']}/{record['total_cases']})"
        print(f"[{index}/{len(java_files)}] {record['name']}: {record['status']}{cases}")

    summary = summarize(records)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )

    print(f"JSONL results: {args.out}")
    print(f"Summary: {args.summary_out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


JAVA_CMD, JAVAC_CMD = 'java', 'javac'


if __name__ == "__main__":
    raise SystemExit(main())
