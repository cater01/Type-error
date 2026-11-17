import argparse
import difflib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from process_guard import JAVA_CMD, JAVAC_CMD, kill_active_processes, run_process



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
# Leave a value as None to use the command-line/default value below it.
JAVA_SOURCES_DIR = ''

TEST_CASE_DIR = ''
# Example:

TEST_RESULTS_JSONL = ''
# Example:


DEFAULT_JAVA_SOURCES_DIR = REPO_ROOT / 'results/generation/avatar/java'
DEFAULT_TEST_CASE_DIR = external_root() / 'avatar/TestCases'
DEFAULT_RESULTS_JSONL = REPO_ROOT / 'results/evaluation/avatar/test_results.jsonl'

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_output(text: str, mode: str) -> str:
    if mode == "strip":
        return text.strip()
    if mode == "rstrip":
        return text.rstrip()
    if mode == "tokens":
        return " ".join(text.split())
    if mode == "exact":
        return text
    raise ValueError(f"Unknown compare mode: {mode}")


def short_diff(expected: str, actual: str, max_lines: int = 80) -> str:
    diff = difflib.unified_diff(
        expected.splitlines(),
        actual.splitlines(),
        fromfile="expected",
        tofile="actual",
        lineterm="",
    )
    return "\n".join(list(diff)[:max_lines])


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


def run_command(
    args: list[str],
    cwd: Path,
    timeout: int,
    input_text: str | None = None,
) -> tuple[str, subprocess.CompletedProcess | None, str, str]:
    process = None
    try:
        process = subprocess.Popen(
            args,
            cwd=cwd,
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate(input=input_text, timeout=timeout)
        return "ok", subprocess.CompletedProcess(args, process.returncode, stdout, stderr), "", ""
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
        return "timeout", None, stdout, stderr
    except KeyboardInterrupt:
        kill_process_tree(process)
        raise
    except Exception as exc:
        kill_process_tree(process)
        return "launch_error", None, "", str(exc)


def get_test_cases_for_file(base_name: str, test_case_dir: Path) -> list[tuple[Path, Path]]:
    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)\.in$")
    test_cases = []
    for in_file in test_case_dir.glob(f"{base_name}_*.in"):
        match = pattern.match(in_file.name)
        if not match:
            continue
        case_num = match.group(1)
        out_file = test_case_dir / f"{base_name}_{case_num}.out"
        if out_file.exists():
            test_cases.append((in_file, out_file))
    test_cases.sort(key=lambda pair: int(pattern.match(pair[0].name).group(1)))
    return test_cases


def compile_java(workdir: Path, timeout: int) -> dict:
    started = time.time()
    status, result, error = run_process(
        [JAVAC_CMD, "-encoding", "UTF-8", "Main.java"],
        cwd=workdir,
        timeout=timeout,
    )
    if status == "timeout":
        return {
            "status": "compile_timeout",
            "returncode": None,
            "stdout": "",
            "stderr": error,
            "elapsed_seconds": round(time.time() - started, 3),
        }
    if status != "ok":
        return {
            "status": "compile_launch_error",
            "returncode": None,
            "stdout": "",
            "stderr": error,
            "elapsed_seconds": round(time.time() - started, 3),
        }
    return {
        "status": "ok" if result.returncode == 0 else "compile_error",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def run_case(workdir: Path, in_file: Path, out_file: Path, timeout: int, compare_mode: str) -> dict:
    started = time.time()
    input_data = read_text(in_file)
    expected_raw = read_text(out_file)

    status, result, error = run_process(
        [JAVA_CMD, "Main"],
        cwd=workdir,
        timeout=timeout,
        input_data=input_data,
    )
    if status == "timeout":
        return {
            "case": in_file.stem,
            "in_file": str(in_file),
            "out_file": str(out_file),
            "status": "runtime_timeout",
            "passed": False,
            "stdout": "",
            "stderr": error,
            "expected": expected_raw,
            "elapsed_seconds": round(time.time() - started, 3),
        }
    if status != "ok":
        return {
            "case": in_file.stem,
            "in_file": str(in_file),
            "out_file": str(out_file),
            "status": "runtime_launch_error",
            "passed": False,
            "stdout": "",
            "stderr": error,
            "expected": expected_raw,
            "elapsed_seconds": round(time.time() - started, 3),
        }

    returncode = result.returncode
    actual = result.stdout
    stderr = result.stderr
    actual_cmp = normalize_output(actual, compare_mode)
    expected_cmp = normalize_output(expected_raw, compare_mode)
    passed = returncode == 0 and actual_cmp == expected_cmp

    status = "passed"
    if returncode != 0:
        status = "runtime_error"
    elif not passed:
        status = "wrong_answer"

    record = {
        "case": in_file.stem,
        "in_file": str(in_file),
        "out_file": str(out_file),
        "status": status,
        "passed": passed,
        "returncode": returncode,
        "stdout": actual,
        "stderr": stderr,
        "expected": expected_raw,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    if status == "wrong_answer":
        record["diff"] = short_diff(expected_cmp, actual_cmp)
    return record


def test_java_file(
    java_file: Path,
    test_case_dir: Path,
    work_root: Path,
    compile_timeout: int,
    run_timeout: int,
    compare_mode: str,
    keep_workdirs: bool,
    fail_fast_cases: bool,
) -> dict:
    started = time.time()
    base_name = java_file.stem
    workdir = work_root / base_name

    record = {
        "name": base_name,
        "java_file": str(java_file),
        "workdir": str(workdir),
        "status": "started",
        "compile": None,
        "cases": [],
        "passed_cases": 0,
        "total_cases": 0,
        "passed": False,
    }

    test_cases = get_test_cases_for_file(base_name, test_case_dir)
    record["total_cases"] = len(test_cases)
    if not test_cases:
        record.update(
            {
                "status": "missing_test_cases",
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
        return record

    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    write_text(workdir / "Main.java", read_text(java_file))

    compile_record = compile_java(workdir, timeout=compile_timeout)
    record["compile"] = compile_record
    if compile_record["status"] != "ok":
        record.update(
            {
                "status": compile_record["status"],
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
        if not keep_workdirs:
            shutil.rmtree(workdir, ignore_errors=True)
        return record

    for in_file, out_file in test_cases:
        case_record = run_case(
            workdir,
            in_file,
            out_file,
            timeout=run_timeout,
            compare_mode=compare_mode,
        )
        record["cases"].append(case_record)
        if case_record["passed"]:
            record["passed_cases"] += 1
        elif fail_fast_cases:
            break

    if record["passed_cases"] == record["total_cases"]:
        record["status"] = "passed"
        record["passed"] = True
    elif any(case["status"].startswith("runtime") for case in record["cases"]):
        record["status"] = "runtime_error"
    else:
        record["status"] = "wrong_answer"

    record["elapsed_seconds"] = round(time.time() - started, 3)
    if not keep_workdirs:
        shutil.rmtree(workdir, ignore_errors=True)
    return record


def summarize(records: list[dict]) -> dict:
    summary = {
        "total_files": len(records),
        "passed_files": 0,
        "compile_errors": 0,
        "runtime_errors": 0,
        "wrong_answer_files": 0,
        "missing_test_cases": 0,
        "passed_cases": 0,
        "total_cases": 0,
    }
    for record in records:
        status = record["status"]
        if status == "passed":
            summary["passed_files"] += 1
        elif status.startswith("compile"):
            summary["compile_errors"] += 1
        elif status == "runtime_error":
            summary["runtime_errors"] += 1
        elif status == "wrong_answer":
            summary["wrong_answer_files"] += 1
        elif status == "missing_test_cases":
            summary["missing_test_cases"] += 1

        summary["passed_cases"] += record.get("passed_cases", 0)
        summary["total_cases"] += record.get("total_cases", 0)

    summary["case_accuracy"] = (
        summary["passed_cases"] / summary["total_cases"]
        if summary["total_cases"]
        else None
    )
    return summary


def main() -> int:
    parser = ConfigArgumentParser(description="Test Avatar Java translations with .in/.out cases.")
    parser.add_argument("--java-dir", type=Path, default=DEFAULT_JAVA_SOURCES_DIR)
    parser.add_argument("--test-case-dir", type=Path, default=DEFAULT_TEST_CASE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS_JSONL)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--compile-timeout", type=int, default=30)
    parser.add_argument("--run-timeout", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--keep-workdirs", action="store_true")
    parser.add_argument(
        "--fail-fast-cases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop testing a file after the first failing case.",
    )
    parser.add_argument(
        "--compare-mode",
        choices=["strip", "rstrip", "tokens", "exact"],
        default="strip",
        help="strip compares output after trimming both ends. tokens ignores all whitespace differences.",
    )
    args = parser.parse_args()
    global JAVA_CMD, JAVAC_CMD
    JAVA_CMD, JAVAC_CMD = configure_java(args)
    if args.test_case_dir == DEFAULT_TEST_CASE_DIR:
        args.test_case_dir = args.external_data_root / 'avatar/TestCases'
    finish_setup(args, args.out.with_suffix('.run.json'), {'java': JAVA_CMD, 'javac': JAVAC_CMD})

    if args.summary_out is None:
        args.summary_out = args.out.with_suffix(".summary.json")

    if not args.java_dir.exists():
        print(f"Java source directory not found: {args.java_dir}")
        return 1
    if not args.test_case_dir.exists():
        print(f"Test case directory not found: {args.test_case_dir}")
        return 1

    java_files = sorted(args.java_dir.glob("*.java"))
    if args.limit is not None:
        java_files = java_files[: args.limit]
    if not java_files:
        print(f"No .java files found in: {args.java_dir}")
        return 1

    if args.out.exists() and not args.append:
        args.out.unlink()
    if args.summary_out.exists() and not args.append:
        args.summary_out.unlink()

    work_root = args.out.parent / ("test_workdirs" if args.keep_workdirs else "test_tmp")
    work_root.mkdir(parents=True, exist_ok=True)

    print(f"Java source directory: {args.java_dir}")
    print(f"Test case directory: {args.test_case_dir}")
    print(f"Results JSONL: {args.out}")
    print(f"Found {len(java_files)} Java files.")

    records = []
    try:
        for index, java_file in enumerate(java_files, start=1):
            record = test_java_file(
                java_file=java_file,
                test_case_dir=args.test_case_dir,
                work_root=work_root,
                compile_timeout=args.compile_timeout,
                run_timeout=args.run_timeout,
                compare_mode=args.compare_mode,
                keep_workdirs=args.keep_workdirs,
                fail_fast_cases=args.fail_fast_cases,
            )
            records.append(record)
            append_jsonl(args.out, record)
            print(
                f"[{index}/{len(java_files)}] {record['name']}: "
                f"{record['status']} ({record['passed_cases']}/{record['total_cases']})"
            )
    finally:
        kill_active_processes()

    summary = summarize(records)
    write_text(args.summary_out, json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary: {args.summary_out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


JAVA_CMD, JAVAC_CMD = 'java', 'javac'


if __name__ == "__main__":
    raise SystemExit(main())
