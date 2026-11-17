import argparse
import difflib
import json
import os
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
# Leave a value as None to use the command-line/default value below it.
JAVA_SOURCES_DIR = ''
# Example:

UNITTEST_DB = ''
# Example:

TEST_RESULTS_JSONL = ''
# Example:


DEFAULT_JAVA_SOURCES_DIR = REPO_ROOT / 'results/generation/xcodeeval/java'
DEFAULT_UNITTEST_DB = external_root() / 'xcodeeval/unittest_db.json'
DEFAULT_RESULTS_JSONL = REPO_ROOT / 'results/evaluation/xcodeeval/test_results.jsonl'

JDK_BIN_DIR = None
JAVA_CMD = 'java'
JAVAC_CMD = 'javac'


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
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    if mode == "strip":
        return "\n".join(line.rstrip() for line in text.split("\n")).strip()
    if mode == "rstrip":
        return "\n".join(line.rstrip() for line in text.split("\n")).rstrip()
    if mode == "tokens":
        return " ".join(text.split())
    if mode == "exact":
        return text
    raise ValueError(f"Unknown compare mode: {mode}")


def expected_outputs(output_val, compare_mode: str) -> list[str]:
    if isinstance(output_val, list):
        return [normalize_output(str(item), compare_mode) for item in output_val]
    return [normalize_output(str(output_val), compare_mode)]


def short_diff(expected: str, actual: str, max_lines: int = 80) -> str:
    diff = difflib.unified_diff(
        expected.splitlines(),
        actual.splitlines(),
        fromfile="expected",
        tofile="actual",
        lineterm="",
    )
    return "\n".join(list(diff)[:max_lines])





def decode_output(data: bytes | str | None) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return data.decode("gbk")
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace")


def extract_src_uid_from_filename(java_file: Path) -> str:
    base = java_file.stem
    if "_" in base:
        return base.rsplit("_", 1)[0]
    return base


def load_unittest_db(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def compile_java(workdir: Path, timeout: int) -> dict:
    started = time.time()
    status, result, stdout, stderr = run_command(
        [JAVAC_CMD, "-encoding", "UTF-8", "Main.java"],
        cwd=workdir,
        timeout=timeout,
    )
    if status == "timeout":
        return {
            "status": "compile_timeout",
            "returncode": None,
            "stdout": stdout,
            "stderr": stderr,
            "elapsed_seconds": round(time.time() - started, 3),
        }
    if status != "ok":
        return {
            "status": "compile_launch_error",
            "returncode": None,
            "stdout": "",
            "stderr": stderr,
            "elapsed_seconds": round(time.time() - started, 3),
        }
    return {
        "status": "ok" if result.returncode == 0 else "compile_error",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def kill_process(proc: subprocess.Popen) -> None:
    kill_process_tree(proc)


def run_case(workdir: Path, case: dict, case_index: int, timeout: int, compare_mode: str) -> dict:
    started = time.time()
    input_str = str(case.get("input", ""))
    if input_str and not input_str.endswith("\n"):
        input_str += "\n"

    expected_raw = case.get("output", "")
    expected_list = expected_outputs(expected_raw, compare_mode)

    proc = None
    try:
        proc = subprocess.Popen(
            [JAVA_CMD, "Main"],
            cwd=workdir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        stdout_bytes, stderr_bytes = proc.communicate(
            input=input_str.encode("utf-8"),
            timeout=timeout,
        )
        stdout = decode_output(stdout_bytes)
        stderr = decode_output(stderr_bytes)
    except subprocess.TimeoutExpired:
        if proc:
            kill_process(proc)
        return {
            "case_index": case_index,
            "status": "runtime_timeout",
            "passed": False,
            "input": input_str,
            "expected": expected_raw,
            "stdout": "",
            "stderr": "",
            "elapsed_seconds": round(time.time() - started, 3),
        }
    except KeyboardInterrupt:
        if proc:
            kill_process(proc)
        raise
    except Exception as exc:
        if proc:
            kill_process(proc)
        return {
            "case_index": case_index,
            "status": "runtime_launch_error",
            "passed": False,
            "input": input_str,
            "expected": expected_raw,
            "stdout": "",
            "stderr": str(exc),
            "elapsed_seconds": round(time.time() - started, 3),
        }

    actual_cmp = normalize_output(stdout, compare_mode)
    passed = proc.returncode == 0 and actual_cmp in expected_list
    status = "passed"
    if proc.returncode != 0:
        status = "runtime_error"
    elif not passed:
        status = "wrong_answer"

    record = {
        "case_index": case_index,
        "status": status,
        "passed": passed,
        "returncode": proc.returncode,
        "input": input_str,
        "expected": expected_raw,
        "expected_normalized": expected_list,
        "stdout": stdout,
        "stderr": stderr,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    if status == "wrong_answer":
        record["actual_normalized"] = actual_cmp
        record["diff"] = short_diff(expected_list[0] if expected_list else "", actual_cmp)
    return record


def test_java_file(
    java_file: Path,
    unittest_db: dict,
    work_root: Path,
    compile_timeout: int,
    run_timeout: int,
    compare_mode: str,
    keep_workdirs: bool,
) -> dict:
    started = time.time()
    result_key = java_file.stem
    src_uid = extract_src_uid_from_filename(java_file)
    workdir = work_root / result_key

    record = {
        "name": result_key,
        "src_uid": src_uid,
        "java_file": str(java_file),
        "workdir": str(workdir),
        "status": "started",
        "compile": None,
        "cases": [],
        "passed_cases": 0,
        "total_cases": 0,
        "passed": False,
    }

    test_cases = unittest_db.get(src_uid)
    if test_cases is None:
        record.update(
            {
                "status": "missing_test_cases",
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
        return record
    if not test_cases:
        record.update(
            {
                "status": "empty_test_cases",
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
        return record

    record["total_cases"] = len(test_cases)
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

    for idx, case in enumerate(test_cases):
        case_record = run_case(
            workdir,
            case,
            case_index=idx,
            timeout=run_timeout,
            compare_mode=compare_mode,
        )
        record["cases"].append(case_record)
        if case_record["passed"]:
            record["passed_cases"] += 1

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
        "empty_test_cases": 0,
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
        elif status == "empty_test_cases":
            summary["empty_test_cases"] += 1
        summary["passed_cases"] += record.get("passed_cases", 0)
        summary["total_cases"] += record.get("total_cases", 0)

    summary["case_accuracy"] = (
        summary["passed_cases"] / summary["total_cases"]
        if summary["total_cases"]
        else None
    )
    return summary


def main() -> int:
    parser = ConfigArgumentParser(description="Test XCodeEval Java translations with unittest_db.json.")
    parser.add_argument("--java-dir", type=Path, default=DEFAULT_JAVA_SOURCES_DIR)
    parser.add_argument("--unittest-db", type=Path, default=DEFAULT_UNITTEST_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS_JSONL)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--compile-timeout", type=int, default=10)
    parser.add_argument("--run-timeout", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--keep-workdirs", action="store_true")
    parser.add_argument(
        "--compare-mode",
        choices=["strip", "rstrip", "tokens", "exact"],
        default="strip",
    )
    args = parser.parse_args()
    global JAVA_CMD, JAVAC_CMD
    JAVA_CMD, JAVAC_CMD = configure_java(args)
    if args.unittest_db == DEFAULT_UNITTEST_DB:
        args.unittest_db = args.external_data_root / 'xcodeeval/unittest_db.json'
    finish_setup(args, args.out.with_suffix('.run.json'), {'java': JAVA_CMD, 'javac': JAVAC_CMD})

    if args.summary_out is None:
        args.summary_out = args.out.with_suffix(".summary.json")

    if not args.java_dir.exists():
        print(f"Java source directory not found: {args.java_dir}")
        return 1
    if not args.unittest_db.exists():
        print(f"unittest_db not found: {args.unittest_db}")
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

    unittest_db = load_unittest_db(args.unittest_db)
    work_root = args.out.parent / ("test_workdirs" if args.keep_workdirs else "test_tmp")
    work_root.mkdir(parents=True, exist_ok=True)

    print(f"Java source directory: {args.java_dir}")
    print(f"unittest_db: {args.unittest_db}")
    print(f"Results JSONL: {args.out}")
    print(f"Found {len(java_files)} Java files.")

    records = []
    for index, java_file in enumerate(java_files, start=1):
        record = test_java_file(
            java_file=java_file,
            unittest_db=unittest_db,
            work_root=work_root,
            compile_timeout=args.compile_timeout,
            run_timeout=args.run_timeout,
            compare_mode=args.compare_mode,
            keep_workdirs=args.keep_workdirs,
        )
        records.append(record)
        append_jsonl(args.out, record)
        print(
            f"[{index}/{len(java_files)}] {record['name']}: "
            f"{record['status']} ({record['passed_cases']}/{record['total_cases']})"
        )

    summary = summarize(records)
    write_text(args.summary_out, json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary: {args.summary_out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


JAVA_CMD, JAVAC_CMD = 'java', 'javac'


if __name__ == "__main__":
    raise SystemExit(main())
