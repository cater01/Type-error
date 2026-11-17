#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 2: post-translation repair prompt experiments with Qwen.

Input JSONL rows should contain:
- source_code: original Python program
- translated_code: incorrect Java translation
- Error message / error_message / failure_feedback / feedback: javac or test failure feedback

Repair strategies:
- r0: ordinary repair prompt
- r1: ordinary repair prompt + known type-error hint
- r2: ordinary repair prompt + four-category type-error repair guidance

This script follows the same style as the pre-translation Qwen enhancement script:
- Qwen chat-template prompt construction
- sampling decoding by default
- temperature=0.7, top_p=0.95, top_k=10
- torch_dtype=model_dtype(TORCH_DTYPE), device_map="auto"
- saves prompts, raw responses, repaired Java code, messages, and JSONL logs
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path



# Shared CLI/config support; works when invoked from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from runtime import (REPO_ROOT, ConfigArgumentParser, generation_datasets,
                     external_root, configure_java, finish_setup, seed_torch,
                     credential, repo_path, model_dtype, matches_error_source)

EXPERIMENT_ROOT = REPO_ROOT
ROOT = REPO_ROOT

DEFAULT_MODEL = os.environ.get('QWEN_MODEL_PATH', 'Qwen/Qwen2.5-Coder-7B-Instruct')
DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

DATASET_DEFAULTS = {name: {'kind': 'repair_jsonl', 'input': REPO_ROOT / 'dataset/type_error_dataset.jsonl', 'out': REPO_ROOT / 'results/rq5/generation' / name} for name in ('magicoder', 'qwen7b', 'qwen32b')}

SOURCE_CODE_FIELDS = ["source_code", "python_code", "python", "src_code", "source"]
BUGGY_JAVA_FIELDS = [
    "translated_code",
    "buggy_java",
    "incorrect_java",
    "java_code",
    "target_code",
    "prediction",
    "translated_java",
]
FEEDBACK_FIELDS = [
    "Error message",
    "error_message",
    "failure_feedback",
    "feedback",
    "compile_error",
    "javac_error",
    "test_feedback",
    "stderr",
    "message",
]
TARGET_FILE_FIELDS = ["target_file", "source_file", "file_name", "filename", "name"]


# =============================================================================
# Basic IO helpers
# =============================================================================

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def first_non_empty(sample: dict, fields: list[str], default: str = "") -> str:
    for field in fields:
        value = sample.get(field)
        if value is not None and str(value).strip():
            return str(value)
    return default


# =============================================================================
# Output cleaning
# =============================================================================

def strip_markdown(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:java)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text


def clean_generated_java(text: str) -> str:
    text = text.strip()

    # If the model echoes markers or section headers, keep only the likely Java part.
    for marker in ["@@ Response", "Repaired Java code:", "Fixed Java code:", "Java code:"]:
        idx = text.find(marker)
        if idx != -1:
            text = text[idx + len(marker):].strip()

    text = strip_markdown(text)

    # Stop if the model begins another artificial section.
    stop_markers = ["@@ Instruction", "@@ Response", "\nOriginal Python", "\nIncorrect Java", "\nFailure feedback"]
    for marker in stop_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].strip()

    return text.strip()


# =============================================================================
# Repair prompt construction
# =============================================================================

R0_REPAIR_GUIDANCE = """You are a skilled software developer proficient in Python and Java.

Your task is to repair an incorrect Java translation using the original Python program, the incorrect Java code, and the failure feedback.

Repair the Java code so that it preserves the behavior of the Python program and resolves the reported failure. Make only necessary changes. Do not add explanations, markdown fences, or extra text.

Return only the full repaired Java code.
"""

R1_REPAIR_GUIDANCE = """You are a skilled software developer proficient in Python and Java.

Your task is to repair an incorrect Java translation using the original Python program, the incorrect Java code, and the failure feedback.

The given Java translation is known to contain a type error. When repairing the code, prioritize fixing type-related problems, including incorrect type choices, inconsistent declarations, incompatible operations between Java types, wrong return types, scalar/container confusion, primitive/reference confusion, char/String confusion, and unsafe or incorrect type conversions.

Repair the Java code so that it preserves the behavior of the Python program and resolves the reported failure.

Return only the full repaired Java code.
"""

R2_REPAIR_GUIDANCE = """You are a skilled software developer proficient in Python and Java.

Your task is to repair an incorrect Java translation using the original Python program, the incorrect Java code, and the failure feedback.

The given Java translation is known to contain a type error. Before writing the repaired Java code, silently check the bug against the following four Python-to-Java type-error categories and fix the most relevant type problem first:

1. Range or precision problems:
   Check whether Java numeric types are too small or imprecise for the Python values. Use long, double, BigInteger, or BigDecimal when int or float would overflow, truncate, or lose precision. Input-reading and parsing methods must match the chosen type.

2. Type declaration inconsistency:
   Check whether Java variable types, method parameter types, return types, array element types, or collection generic types are inconsistent with the Python value semantics or the surrounding Java context. Keep scalar values, booleans, numbers, chars, strings, arrays, lists, sets, maps, and iterables distinct.

3. Type-incompatible operations:
   Check whether the Java code applies operations that are illegal or semantically inappropriate for the chosen Java type. Do not treat arrays as Lists, strings as mutable arrays, sets as maps, maps as lists, or scalar values as containers. Preserve Python list, tuple, set, dict, string, and iterable semantics with suitable Java structures.

4. Type conversion errors:
   Check whether explicit or implicit conversions were translated incorrectly. Avoid unsafe narrowing conversions, premature casts, invalid casts, integer division when Python requires floating-point division, incorrect rounding, boxing/unboxing mistakes, and char/String confusion.

Repair the Java code so that it preserves the behavior of the Python program and resolves the reported failure. Make only necessary changes. Do not add explanations, markdown fences, or extra text.

Return only the full repaired Java code.
"""


def repair_guidance(repair_style: str) -> str:
    if repair_style == "r0":
        return R0_REPAIR_GUIDANCE
    if repair_style == "r1":
        return R1_REPAIR_GUIDANCE
    if repair_style == "r2":
        return R2_REPAIR_GUIDANCE
    raise ValueError(f"Unknown repair style: {repair_style}")


def build_repair_prompt(source_code: str, buggy_java: str, feedback: str, repair_style: str) -> str:
    guidance = repair_guidance(repair_style)
    return f"""{guidance}

Original Python program:
```python
{source_code}
```

Incorrect Java translation:
```java
{buggy_java}
```

Failure feedback:
```text
{feedback}
```
"""


def build_messages(user_prompt: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# =============================================================================
# Qwen loading and generation
# =============================================================================

def load_model(model_path: str):
    seed_torch()
    try:
        from modelscope import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "The ModelScope package is required. Install it in your Qwen environment, "
            "for example: pip install modelscope"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_dtype(TORCH_DTYPE),
        device_map="auto",
    )
    model.eval()

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Model loaded.")
    print(f"Model dtype: {getattr(model, 'dtype', 'unknown')}")
    print(f"Model device: {getattr(model, 'device', 'unknown')}")
    return model, tokenizer


def generate_java(
    model,
    tokenizer,
    user_prompt: str,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    max_input_tokens: int | None,
) -> str:
    import torch

    messages = build_messages(user_prompt, system_prompt)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Keep the same style as the pre-translation script.
    input_device = next(model.parameters()).device
    tokenize_kwargs = {"return_tensors": "pt"}
    if max_input_tokens is not None:
        tokenize_kwargs.update({"truncation": True, "max_length": max_input_tokens})
    model_inputs = tokenizer([text], **tokenize_kwargs).to(input_device)

    generate_kwargs = {
        **model_inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs.update({
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        })

    with torch.no_grad():
        generated_ids = model.generate(**generate_kwargs)

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


# =============================================================================
# Java compilation and optional tests
# =============================================================================

def normalize_target_file(target_file: str, line_idx: int) -> str:
    target_file = Path(str(target_file)).name.strip()
    if not target_file:
        target_file = f"sample_{line_idx}.java"
    if not target_file.endswith(".java"):
        target_file = Path(target_file).with_suffix(".java").name
    return target_file


def compile_java(java_code: str, target_file: str, timeout: int = 20) -> dict:
    if shutil.which(JAVAC_CMD) is None:
        return {
            "compile_status": "javac_not_found",
            "compile_pass": None,
            "compile_stdout": "",
            "compile_stderr": "javac not found in PATH",
        }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        java_path = tmp_path / target_file
        write_text(java_path, java_code)

        proc = subprocess.run(
            [JAVAC_CMD, "-encoding", "UTF-8", java_path.name],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "compile_status": "ok" if proc.returncode == 0 else "compile_error",
            "compile_pass": proc.returncode == 0,
            "compile_stdout": proc.stdout,
            "compile_stderr": proc.stderr,
        }


def extract_main_class(java_code: str, target_file: str) -> str:
    # Prefer class containing main method.
    main_match = re.search(
        r"class\s+([A-Za-z_][A-Za-z0-9_]*)[\s\S]*?public\s+static\s+void\s+main\s*\(",
        java_code,
    )
    if main_match:
        return main_match.group(1)

    public_match = re.search(r"public\s+class\s+([A-Za-z_][A-Za-z0-9_]*)", java_code)
    if public_match:
        return public_match.group(1)

    class_match = re.search(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)", java_code)
    if class_match:
        return class_match.group(1)

    return Path(target_file).stem


def normalize_output(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()


def run_tests_if_present(java_code: str, target_file: str, test_cases, timeout: int = 20) -> dict:
    if not test_cases:
        return {
            "test_status": "no_tests",
            "test_pass": None,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_failures": [],
        }

    if shutil.which(JAVAC_CMD) is None or shutil.which(JAVA_CMD) is None:
        return {
            "test_status": "java_tools_not_found",
            "test_pass": None,
            "total_tests": len(test_cases),
            "passed_tests": 0,
            "failed_tests": len(test_cases),
            "test_failures": [{"error": "javac or java not found in PATH"}],
        }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        java_path = tmp_path / target_file
        write_text(java_path, java_code)

        compile_proc = subprocess.run(
            [JAVAC_CMD, "-encoding", "UTF-8", java_path.name],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        if compile_proc.returncode != 0:
            return {
                "test_status": "compile_error",
                "test_pass": False,
                "total_tests": len(test_cases),
                "passed_tests": 0,
                "failed_tests": len(test_cases),
                "test_failures": [{
                    "stage": "compile",
                    "stderr": compile_proc.stderr,
                    "stdout": compile_proc.stdout,
                }],
            }

        main_class = extract_main_class(java_code, target_file)
        passed = 0
        failures = []

        for idx, tc in enumerate(test_cases):
            if isinstance(tc, dict):
                inp = str(tc.get("input", ""))
                expected = str(tc.get("output", tc.get("expected", "")))
            elif isinstance(tc, (list, tuple)) and len(tc) >= 2:
                inp = str(tc[0])
                expected = str(tc[1])
            else:
                failures.append({"index": idx, "error": "Unsupported test case format", "test_case": tc})
                continue

            proc = subprocess.run(
                [JAVA_CMD, main_class],
                cwd=tmp_path,
                input=inp,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
            actual = proc.stdout
            ok = proc.returncode == 0 and normalize_output(actual) == normalize_output(expected)
            if ok:
                passed += 1
            else:
                failures.append({
                    "index": idx,
                    "returncode": proc.returncode,
                    "input": inp,
                    "expected": expected,
                    "actual": actual,
                    "stderr": proc.stderr,
                })

        total = len(test_cases)
        return {
            "test_status": "ok",
            "test_pass": passed == total,
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": total - passed,
            "test_failures": failures[:5],
        }


def get_test_cases(sample: dict):
    for field in ["test_cases", "tests", "io_tests", "examples"]:
        value = sample.get(field)
        if isinstance(value, list):
            return value
    return []


# =============================================================================
# Repair experiment loop
# =============================================================================

def repair_error_jsonl(args, dataset: str, jsonl_path: Path, out_dir: Path, model, tokenizer) -> int:
    repaired_dir = out_dir / "repaired_java"
    raw_dir = out_dir / "raw_responses"
    prompt_dir = out_dir / "prompts"
    messages_dir = out_dir / "messages"
    result_jsonl = out_dir / "repair_results.jsonl"
    compile_jsonl = out_dir / "compile_results.jsonl"
    summary_json = out_dir / "summary.json"

    if not jsonl_path.exists():
        print(f"Input JSONL not found: {jsonl_path}")
        return 1

    processed = 0
    ok_count = 0
    compile_total = 0
    compile_pass_count = 0
    tests_total_samples = 0
    tests_pass_samples = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if args.end_idx is not None and line_idx >= args.end_idx:
                break
            if line_idx < args.start_idx:
                continue
            if args.limit is not None and processed >= args.limit:
                break
            if not line.strip():
                continue

            started = time.time()
            record = {
                "line_idx": line_idx,
                "id": None,
                "model_dataset": dataset,
                "repair_style": args.repair_style,
                "model_path": args.model,
                "decoding": "sampling" if args.do_sample else "greedy",
                "temperature": args.temperature if args.do_sample else None,
                "top_p": args.top_p if args.do_sample else None,
                "top_k": args.top_k if args.do_sample else None,
                "max_new_tokens": args.max_new_tokens,
                "status": "started",
            }

            try:
                sample = json.loads(line)
                if not matches_error_source(sample, dataset):
                    continue
                record["id"] = sample.get("id")
                record["source_file"] = sample.get("source_file")
                record["label_0"] = sample.get("label_0")
                record["label_1"] = sample.get("label_1")
                record["label_source"] = sample.get("label_source")
                sample_dataset = str(sample.get("dataset", "unknown"))
                record["sample_dataset"] = sample_dataset

                source_code = first_non_empty(sample, SOURCE_CODE_FIELDS).strip()
                buggy_java = first_non_empty(sample, BUGGY_JAVA_FIELDS).strip()
                feedback = first_non_empty(sample, FEEDBACK_FIELDS, default="No failure feedback provided.").strip()

                if not source_code:
                    raise ValueError("Empty source code in JSONL row")
                if not buggy_java:
                    raise ValueError("Empty translated_code / buggy Java in JSONL row")

                source_bytes = len(source_code.encode("utf-8"))
                record["source_size_bytes"] = source_bytes
                if source_bytes > args.max_source_size_bytes:
                    record.update({
                        "status": "skipped_too_large",
                        "max_source_size_bytes": args.max_source_size_bytes,
                        "elapsed_seconds": round(time.time() - started, 3),
                    })
                    append_jsonl(result_jsonl, record)
                    print(f"[{line_idx}] skip too large source: {source_bytes} bytes")
                    continue

                raw_target = first_non_empty(sample, TARGET_FILE_FIELDS, default=f"sample_{line_idx}.java")
                target_file = normalize_target_file(raw_target, line_idx)
                output_stem = safe_filename(Path(target_file).stem)

                output_path = repaired_dir / sample_dataset / target_file
                raw_path = raw_dir / f"{line_idx:05d}_{output_stem}.txt"
                prompt_path = prompt_dir / f"{line_idx:05d}_{output_stem}.txt"
                message_path = messages_dir / f"{line_idx:05d}_{output_stem}.json"

                record.update({
                    "target_file": target_file,
                    "repaired_java_path": str(output_path),
                    "raw_response_path": str(raw_path),
                    "prompt_path": str(prompt_path),
                    "messages_path": str(message_path),
                })

                if args.resume and output_path.exists():
                    record["status"] = "skipped_existing"
                    append_jsonl(result_jsonl, record)
                    print(f"[{processed + 1}] line {line_idx}: skip existing {target_file}")
                    continue

                prompt = build_repair_prompt(source_code, buggy_java, feedback, args.repair_style)
                messages = build_messages(prompt, args.system_prompt)

                print(f"[{processed + 1}] line {line_idx}: repairing {target_file}")
                raw = generate_java(
                    model,
                    tokenizer,
                    prompt,
                    args.system_prompt,
                    args.max_new_tokens,
                    args.temperature,
                    args.top_p,
                    args.top_k,
                    args.do_sample,
                    args.max_input_tokens,
                )
                repaired_java = clean_generated_java(raw)
                if not repaired_java:
                    repaired_java = "// Warning: No Java code could be extracted from the model output."

                write_text(output_path, repaired_java)
                write_text(raw_path, raw)
                write_text(prompt_path, prompt)
                write_text(message_path, json.dumps(messages, ensure_ascii=False, indent=2))

                record.update({
                    "status": "ok",
                    "elapsed_seconds": round(time.time() - started, 3),
                })

                ok_count += 1
                processed += 1

                compile_record = {
                    "line_idx": line_idx,
                    "id": sample.get("id"),
                    "dataset": dataset,
                    "sample_dataset": sample_dataset,
                    "repair_style": args.repair_style,
                    "target_file": target_file,
                    "repaired_java_path": str(output_path),
                }

                if args.run_javac:
                    compile_result = compile_java(repaired_java, target_file, timeout=args.eval_timeout)
                    compile_record.update(compile_result)
                    compile_total += 1
                    if compile_result.get("compile_pass") is True:
                        compile_pass_count += 1
                else:
                    compile_record.update({
                        "compile_status": "not_run",
                        "compile_pass": None,
                        "compile_stdout": "",
                        "compile_stderr": "",
                    })

                test_cases = get_test_cases(sample)
                if args.run_tests and test_cases:
                    test_result = run_tests_if_present(repaired_java, target_file, test_cases, timeout=args.eval_timeout)
                    compile_record.update(test_result)
                    tests_total_samples += 1
                    if test_result.get("test_pass") is True:
                        tests_pass_samples += 1
                else:
                    compile_record.update({
                        "test_status": "no_tests" if not test_cases else "not_run",
                        "test_pass": None,
                        "total_tests": len(test_cases),
                        "passed_tests": 0,
                        "failed_tests": 0,
                        "test_failures": [],
                    })

                append_jsonl(compile_jsonl, compile_record)

            except Exception as exc:
                record.update({
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds": round(time.time() - started, 3),
                })
                print(f"  error at line {line_idx}: {type(exc).__name__}: {exc}", file=sys.stderr)

            append_jsonl(result_jsonl, record)

    summary = {
        "dataset": dataset,
        "repair_style": args.repair_style,
        "model_path": args.model,
        "processed": processed,
        "ok": ok_count,
        "compile_total": compile_total,
        "compile_pass": compile_pass_count,
        "compile_pass_rate": (compile_pass_count / compile_total) if compile_total else None,
        "test_total_samples": tests_total_samples,
        "test_pass_samples": tests_pass_samples,
        "test_pass_rate": (tests_pass_samples / tests_total_samples) if tests_total_samples else None,
        "repair_results": str(result_jsonl),
        "compile_results": str(compile_jsonl),
    }
    write_text(summary_json, json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"Done. Repaired {processed} samples.")
    print(f"Repair log: {result_jsonl}")
    print(f"Compile/test log: {compile_jsonl}")
    print(f"Summary: {summary_json}")
    print(f"Compile pass rate: {summary['compile_pass_rate']}")
    print(f"Test pass rate: {summary['test_pass_rate']}")
    return 0


def run_one(args, dataset: str, repair_style: str, model, tokenizer) -> int:
    defaults = DATASET_DEFAULTS[dataset]
    input_path = args.input or defaults["input"]

    # Use a separate output directory for each repair strategy.
    base_out = args.out_dir or defaults["out"]
    out_dir = base_out / repair_style

    # Create a shallow copy-like namespace so args.repair_style can vary in "all".
    local_args = argparse.Namespace(**vars(args))
    local_args.repair_style = repair_style

    print("=" * 80)
    print(f"Dataset: {dataset}")
    print(f"Repair style: {repair_style}")
    print(f"Input: {input_path}")
    print(f"Output: {out_dir}")
    print(f"Decoding: {'sampling' if local_args.do_sample else 'greedy'}")
    print("=" * 80)

    return repair_error_jsonl(local_args, dataset, input_path, out_dir, model, tokenizer)


def main() -> int:
    parser = ConfigArgumentParser(description="Run post-translation repair prompt experiments with Qwen.")
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), required=True)
    parser.add_argument("--repair-style", choices=["r0", "r1", "r2", "all"], required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input", type=Path, default=None, help="JSONL repair dataset.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)

    # Keep defaults consistent with the pre-translation Qwen script.
    parser.add_argument("--max-source-size-bytes", type=int, default=5 * 1024)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)

    parser.add_argument("--run-javac", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-javac", action="store_true", help="Disable javac compilation check.")
    parser.add_argument("--run-tests", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-tests", action="store_true", help="Disable test execution even if test cases exist.")
    parser.add_argument("--eval-timeout", type=int, default=20)

    parser.add_argument('--model-path', type=Path, default=None, help='Local model directory, resolved relative to the repository; overrides --model.')
    parser.add_argument('--model-revision', default=None, help='Exact model revision when known; unset preserves the original loader behavior.')
    parser.add_argument('--prompt-bundle', type=Path, default=None, help='Optional frozen type-hint bundle for prevention experiments.')
    parser.add_argument('--torch-dtype', choices=['auto', 'float16', 'bfloat16', 'float32'], default='auto')
    args = parser.parse_args()
    configure_generation(args)


    if args.greedy:
        args.do_sample = False
    if args.no_javac:
        args.run_javac = False
    if args.no_tests:
        args.run_tests = False

    selected_styles = ["r0", "r1", "r2"] if args.repair_style == "all" else [args.repair_style]

    print(f"Selected dataset: {args.dataset}")
    print(f"Selected repair styles: {selected_styles}")
    print(f"Model: {args.model}")
    print(f"Decoding: {'sampling' if args.do_sample else 'greedy'}")

    model, tokenizer = load_model(args.model)

    exit_code = 0
    for style in selected_styles:
        code = run_one(args, args.dataset, style, model, tokenizer)
        exit_code = max(exit_code, code)

    print("All repair-style runs finished.")
    return exit_code



MODEL_REVISION = None
TORCH_DTYPE = 'auto'
JAVA_CMD, JAVAC_CMD = 'java', 'javac'

def configure_generation(args):
    global DATASET_DEFAULTS, MODEL_REVISION, TORCH_DTYPE, JAVA_CMD, JAVAC_CMD
    if args.model_path is not None:
        args.model = str(args.model_path)
    DATASET_DEFAULTS = {name: {'kind': 'repair_jsonl', 'input': REPO_ROOT / 'dataset/type_error_dataset.jsonl', 'out': REPO_ROOT / 'results/rq5/generation' / name} for name in ('magicoder', 'qwen7b', 'qwen32b')}
    if args.prompt_bundle:
        raise ValueError('--prompt-bundle is a prevention-only option')
    MODEL_REVISION, TORCH_DTYPE = args.model_revision, args.torch_dtype
    JAVA_CMD, JAVAC_CMD = configure_java(args)
    args.input = args.input or DATASET_DEFAULTS[args.dataset]['input']
    args.out_dir = args.out_dir or DATASET_DEFAULTS[args.dataset]['out']
    finish_setup(args, args.out_dir / f'run_config_{args.repair_style}.json', {'java': JAVA_CMD, 'javac': JAVAC_CMD})


if __name__ == "__main__":
    raise SystemExit(main())
