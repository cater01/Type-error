import argparse
import json
import os
import re
import sys
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
DESKTOP = REPO_ROOT
DEFAULT_MODEL = os.environ.get('QWEN_MODEL_PATH', 'Qwen/Qwen2.5-Coder-7B-Instruct')
DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


DATASET_DEFAULTS = generation_datasets()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def strip_markdown(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:java)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text


def clean_generated_java(text: str) -> str:
    text = strip_markdown(text)
    if "@@" in text:
        text = text.split("@@", 1)[0].strip()
    return text.strip()


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def remove_javafx_import(java_code: str) -> str:
    return re.sub(r"(?m)^\s*import\s+javafx\.util\.Pair\s*;\s*\n?", "", java_code)


def discover_transcoder_names(py_dir: Path, test_dir: Path) -> list[str]:
    py_names = {p.stem for p in py_dir.glob("*.py")}
    test_names = {p.stem for p in test_dir.glob("*.java")}
    return sorted(py_names & test_names)


def extract_f_gold_signature(template: str) -> str:
    match = re.search(
        r"(?m)^\s*(?:public\s+)?static\s+(.+?)\s+f_gold\s*\((.*?)\)\s*\{",
        template,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("Could not find f_gold signature in test template")
    return_type = " ".join(match.group(1).split())
    params = " ".join(match.group(2).split())
    return f"static {return_type} f_filled ( {params} )"


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

    raise ValueError("Unbalanced braces in generated method")


def parse_signature(signature: str) -> tuple[str, list[str]]:
    match = re.search(r"static\s+(.+?)\s+f_filled\s*\((.*)\)\s*$", signature, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse expected signature: {signature}")
    return_type = " ".join(match.group(1).split())
    params_text = match.group(2).strip()
    if not params_text:
        return return_type, []
    return return_type, [part.strip() for part in params_text.split(",")]


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
                "start": match.start(),
                "end": close_brace + 1,
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


def extract_method(java_text: str, expected_signature: str) -> str:
    java_text = strip_markdown(java_text).replace("@@ Response", "").strip()
    methods = java_methods(java_text)
    if not methods:
        raise ValueError("No Java method found in model output")

    expected_return_type, expected_params = parse_signature(expected_signature)
    method = max(methods, key=lambda item: method_score(item, expected_return_type, expected_params))
    if method_score(method, expected_return_type, expected_params) < 0:
        raise ValueError("No suitable Java method found in model output")

    original_method_name = method["name"]
    body = method["text"][method["text"].find("{") :].strip()
    if original_method_name != "f_filled":
        body = re.sub(rf"\b{re.escape(original_method_name)}\s*\(", "f_filled(", body)
    return f"{expected_signature} {body}"


def type_hint_block(prompt_style: str) -> str:
    if PROMPT_HINTS_OVERRIDE is not None:
        return PROMPT_HINTS_OVERRIDE[prompt_style]
    if prompt_style == "p0":
        return ""
    if prompt_style == "p1":
        return """
Additional requirement:
Pay special attention to Python-to-Java type differences. Avoid type errors in the Java output, including integer overflow, floating-point precision loss, incompatible primitive/reference types, invalid operations on strings/chars/arrays/lists, and unsafe type conversions.
"""
    if prompt_style == "p2":
        return """
Additional type-error prevention checklist:
Before writing the Java output, silently check the translation against these four Python-to-Java type error categories. The examples below are abstracted from the manually labeled type-error dataset:

1. Range or precision problems:
    Do not mechanically use int or float when Python values may require a larger range or higher precision. Use long, double, BigInteger, or BigDecimal when needed, and make input reading or parsing consistent with the selected type.

2. Type declaration inconsistency:
  Keep Java variable types, method parameter types, return types, array element types, and collection generic types consistent with the actual Python value semantics. Distinguish scalars from containers, booleans from arrays, strings from numbers, and lists, sets, maps, tuples, and iterables from one another.

3. Type-incompatible operations:
   Choose Java types that support the operations required by the Python code. Do not treat arrays as Lists, strings as numeric or mutable arrays, sets as maps, maps as lists, or scalar values as containers.
4. Type conversion errors:
  Preserve Python’s explicit and implicit conversion semantics. Avoid premature casts, unsafe narrowing conversions, integer division when Python requires floating-point division, incorrect rounding, invalid casts, boxing/unboxing mistakes, and char/String confusion. Perform arithmetic first and cast or round only when it matches the Python behavior.characters.

Return only the final Java code after applying this checklist.
"""
    raise ValueError(f"Unknown prompt style: {prompt_style}")


def build_program_prompt(python_code: str, dataset: str, prompt_style: str) -> str:
    extra = type_hint_block(prompt_style)
    base = """You are a skilled software developer proficient in multiple programming languages.
Your task is to re-write the input source code.
Below is the input source code written in Python that you should re-write into Java programming language.
You must respond with the Java output code only.
"""
    if prompt_style == "p0":
        return f"""{base}
{python_code}
"""
    return f"""You are a skilled software developer proficient in multiple programming languages.
Your task is to re-write the input source code.
Below is the input source code written in Python that you should re-write into Java programming language.
You must respond with the Java output code only.
{extra}
{python_code}
"""


def build_transcoder_prompt(python_code: str, expected_signature: str, prompt_style: str) -> str:
    extra = type_hint_block(prompt_style)
    return f"""You are translating a TransCoder benchmark function from Python to Java.

Return only one Java method. Do not return a class, main method, markdown, or explanation.
The Java method signature must be exactly:
{expected_signature}

Use the method parameters directly. Do not create sample inputs, Scanner, System.out.print, or a main method.
If the Python function is recursive, recursive calls in Java must call f_filled.
If helper logic is needed, inline it inside f_filled unless it is impossible.
{extra}
Translate the Python function below so that it matches the behavior exactly:

{python_code}
"""


def build_messages(user_prompt: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


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
    input_device = next(model.parameters()).device
    tokenize_kwargs = {"return_tensors": "pt"}
    if max_input_tokens is not None:
        tokenize_kwargs.update({"truncation": True, "max_length": max_input_tokens})
    model_inputs = tokenizer([text], **tokenize_kwargs).to(input_device)

    generate_kwargs = {
        **model_inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs.update({"temperature": temperature, "top_p": top_p, "top_k": top_k})

    with torch.no_grad():
        generated_ids = model.generate(**generate_kwargs)

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def translate_dir_dataset(args, dataset: str, input_dir: Path, out_dir: Path, model, tokenizer) -> int:
    java_dir = out_dir / "java"
    raw_dir = out_dir / "raw_responses"
    prompt_dir = out_dir / "prompts"
    messages_dir = out_dir / "messages"
    result_jsonl = out_dir / "translation_results.jsonl"

    python_files = sorted(input_dir.glob("*.py"))
    if args.limit is not None:
        python_files = python_files[: args.limit]
    if not python_files:
        print(f"No Python files found: {input_dir}")
        return 1

    for index, py_path in enumerate(python_files, start=1):
        started = time.time()
        name = py_path.stem
        java_path = java_dir / f"{name}.java"
        record = {"dataset": dataset, "prompt_style": args.prompt_style, "name": name, "python_path": str(py_path), "java_path": str(java_path), "status": "started"}

        try:
            if args.resume and java_path.exists():
                record["status"] = "skipped_existing"
                append_jsonl(result_jsonl, record)
                print(f"[{index}/{len(python_files)}] skip existing {py_path.name}")
                continue

            file_size = py_path.stat().st_size
            record["file_size_bytes"] = file_size
            if file_size > args.max_file_size_bytes:
                record.update({"status": "skipped_too_large", "max_file_size_bytes": args.max_file_size_bytes})
                append_jsonl(result_jsonl, record)
                print(f"[{index}/{len(python_files)}] skip too large {py_path.name}: {file_size} bytes")
                continue

            python_code = read_text(py_path)
            prompt = build_program_prompt(python_code, dataset, args.prompt_style)
            messages = build_messages(prompt, args.system_prompt)
            print(f"[{index}/{len(python_files)}] translating {py_path.name}")
            raw = generate_java(model, tokenizer, prompt, args.system_prompt, args.max_new_tokens, args.temperature, args.top_p, args.top_k, args.do_sample, args.max_input_tokens)
            java_code = clean_generated_java(raw) or "// Warning: No Java code could be extracted from the model output."

            write_text(java_path, java_code)
            write_text(raw_dir / f"{name}.txt", raw)
            write_text(prompt_dir / f"{name}.txt", prompt)
            write_text(messages_dir / f"{name}.json", json.dumps(messages, ensure_ascii=False, indent=2))
            record.update({"status": "ok", "raw_response_path": str(raw_dir / f"{name}.txt"), "prompt_path": str(prompt_dir / f"{name}.txt"), "messages_path": str(messages_dir / f"{name}.json"), "elapsed_seconds": round(time.time() - started, 3)})
        except Exception as exc:
            record.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc), "elapsed_seconds": round(time.time() - started, 3)})
            print(f"  error: {type(exc).__name__}: {exc}", file=sys.stderr)
        append_jsonl(result_jsonl, record)
    print(f"Translation log: {result_jsonl}")
    return 0


def translate_xcodeeval(args, jsonl_path: Path, out_dir: Path, model, tokenizer) -> int:
    java_dir = out_dir / "java"
    raw_dir = out_dir / "raw_responses"
    prompt_dir = out_dir / "prompts"
    messages_dir = out_dir / "messages"
    result_jsonl = out_dir / "translation_results.jsonl"
    if not jsonl_path.exists():
        print(f"Input JSONL not found: {jsonl_path}")
        return 1

    uid_counter = {}
    processed = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if args.end_idx is not None and line_idx >= args.end_idx:
                break
            if line_idx < args.start_idx:
                continue
            if args.limit is not None and processed >= args.limit:
                break

            started = time.time()
            record = {"dataset": "xcodeeval", "prompt_style": args.prompt_style, "line_idx": line_idx, "status": "started"}
            try:
                sample = json.loads(line)
                src_uid = str(sample[args.uid_field])
                python_code = str(sample[args.source_field]).strip()
                uid_counter[src_uid] = uid_counter.get(src_uid, 0) + 1
                output_stem = f"{safe_filename(src_uid)}_{uid_counter[src_uid]}"
                java_path = java_dir / f"{output_stem}.java"
                record.update({"src_uid": src_uid, "output_stem": output_stem, "java_path": str(java_path)})

                if args.resume and java_path.exists():
                    record["status"] = "skipped_existing"
                    append_jsonl(result_jsonl, record)
                    print(f"[{processed + 1}] line {line_idx}: skip existing {java_path.name}")
                    continue

                prompt = build_program_prompt(python_code, "xcodeeval", args.prompt_style)
                messages = build_messages(prompt, args.system_prompt)
                raw = generate_java(model, tokenizer, prompt, args.system_prompt, args.max_new_tokens, args.temperature, args.top_p, args.top_k, args.do_sample, args.max_input_tokens)
                java_code = clean_generated_java(raw) or "// Warning: No Java code could be extracted from the model output."

                write_text(java_path, java_code)
                write_text(raw_dir / f"{output_stem}.txt", raw)
                write_text(prompt_dir / f"{output_stem}.txt", prompt)
                write_text(messages_dir / f"{output_stem}.json", json.dumps(messages, ensure_ascii=False, indent=2))
                record.update({"status": "ok", "raw_response_path": str(raw_dir / f"{output_stem}.txt"), "prompt_path": str(prompt_dir / f"{output_stem}.txt"), "messages_path": str(messages_dir / f"{output_stem}.json"), "elapsed_seconds": round(time.time() - started, 3)})
                processed += 1
                print(f"[{processed}] line {line_idx}: saved {java_path.name}")
            except Exception as exc:
                record.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc), "elapsed_seconds": round(time.time() - started, 3)})
                print(f"  error at line {line_idx}: {type(exc).__name__}: {exc}", file=sys.stderr)
            append_jsonl(result_jsonl, record)
    print(f"Done. Translated {processed} samples.")
    print(f"Translation log: {result_jsonl}")
    return 0


def translate_transcoder(args, python_dir: Path, test_dir: Path, out_dir: Path, model, tokenizer) -> int:
    prompt_dir = out_dir / "prompts"
    messages_dir = out_dir / "messages"
    raw_dir = out_dir / "raw_responses"
    method_dir = out_dir / "methods"
    filled_dir = out_dir / "filled_java"
    result_jsonl = out_dir / "translation_results.jsonl"

    names = discover_transcoder_names(python_dir, test_dir)
    if args.limit is not None:
        names = names[: args.limit]
    if not names:
        print(f"No matching TransCoder files found: {python_dir} / {test_dir}")
        return 1

    for index, name in enumerate(names, start=1):
        started = time.time()
        py_path = python_dir / f"{name}.py"
        template_path = test_dir / f"{name}.java"
        filled_path = filled_dir / f"{name}.java"
        record = {"dataset": "transcoder", "prompt_style": args.prompt_style, "name": name, "python_path": str(py_path), "test_template_path": str(template_path), "filled_java_path": str(filled_path), "status": "started"}

        try:
            if args.resume and filled_path.exists():
                record["status"] = "skipped_existing"
                append_jsonl(result_jsonl, record)
                print(f"[{index}/{len(names)}] skip existing {name}")
                continue

            python_code = read_text(py_path)
            template = read_text(template_path)
            expected_signature = extract_f_gold_signature(template)
            prompt = build_transcoder_prompt(python_code, expected_signature, args.prompt_style)
            messages = build_messages(prompt, args.system_prompt)

            print(f"[{index}/{len(names)}] translating {name}")
            raw = generate_java(model, tokenizer, prompt, args.system_prompt, args.max_new_tokens, args.temperature, args.top_p, args.top_k, args.do_sample, args.max_input_tokens)
            method = extract_method(raw, expected_signature)
            filled = remove_javafx_import(template).replace("//TOFILL", method)

            write_text(prompt_dir / f"{name}.txt", prompt)
            write_text(messages_dir / f"{name}.json", json.dumps(messages, ensure_ascii=False, indent=2))
            write_text(raw_dir / f"{name}.txt", raw)
            write_text(method_dir / f"{name}.java", method + "\n")
            write_text(filled_path, filled)
            record.update({"status": "ok", "expected_signature": expected_signature, "raw_response_path": str(raw_dir / f"{name}.txt"), "prompt_path": str(prompt_dir / f"{name}.txt"), "messages_path": str(messages_dir / f"{name}.json"), "method_path": str(method_dir / f"{name}.java"), "elapsed_seconds": round(time.time() - started, 3)})
        except Exception as exc:
            record.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc), "elapsed_seconds": round(time.time() - started, 3)})
            print(f"  error: {type(exc).__name__}: {exc}", file=sys.stderr)
        append_jsonl(result_jsonl, record)
    print(f"Translation log: {result_jsonl}")
    return 0


def translate_error_jsonl(args, jsonl_path: Path, out_dir: Path, model, tokenizer) -> int:
    java_dir = out_dir / "java"
    raw_dir = out_dir / "raw_responses"
    prompt_dir = out_dir / "prompts"
    messages_dir = out_dir / "messages"
    result_jsonl = out_dir / "translation_results.jsonl"

    if not jsonl_path.exists():
        print(f"Input JSONL not found: {jsonl_path}")
        return 1

    processed = 0
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
            sample = json.loads(line)
            if not matches_error_source(sample, args.dataset):
                continue
            source_code = str(sample.get(args.source_field, "")).strip()
            target_file = str(sample.get("target_file") or sample.get("source_file") or f"sample_{line_idx}.java")
            if not target_file.endswith(".java"):
                target_file = Path(target_file).with_suffix(".java").name
            output_stem = Path(target_file).stem
            sample_dataset = str(sample.get("dataset", "unknown"))

            record = {
                "line_idx": line_idx,
                "id": sample.get("id"),
                "model_dataset": args.dataset,
                "sample_dataset": sample_dataset,
                "prompt_style": args.prompt_style,
                "source_file": sample.get("source_file"),
                "target_file": target_file,
                "label_0": sample.get("label_0"),
                "label_1": sample.get("label_1"),
                "label_source": sample.get("label_source"),
                "status": "started",
            }

            try:
                if not source_code:
                    raise ValueError("Empty source_code in JSONL row")

                output_path = java_dir / sample_dataset / target_file
                record["java_path"] = str(output_path)
                if args.resume and output_path.exists():
                    record["status"] = "skipped_existing"
                    append_jsonl(result_jsonl, record)
                    print(f"[{processed + 1}] line {line_idx}: skip existing {target_file}")
                    continue

                prompt = build_program_prompt(source_code, sample_dataset.lower(), args.prompt_style)
                messages = build_messages(prompt, args.system_prompt)
                print(f"[{processed + 1}] line {line_idx}: translating {target_file}")
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

                java_code = clean_generated_java(raw) or "// Warning: No Java code could be extracted from the model output."
                write_text(output_path, java_code)
                write_text(raw_dir / f"{output_stem}.txt", raw)
                write_text(prompt_dir / f"{output_stem}.txt", prompt)
                write_text(messages_dir / f"{output_stem}.json", json.dumps(messages, ensure_ascii=False, indent=2))
                record.update(
                    {
                        "status": "ok",
                        "raw_response_path": str(raw_dir / f"{output_stem}.txt"),
                        "prompt_path": str(prompt_dir / f"{output_stem}.txt"),
                        "messages_path": str(messages_dir / f"{output_stem}.json"),
                        "elapsed_seconds": round(time.time() - started, 3),
                    }
                )
                processed += 1
            except Exception as exc:
                record.update(
                    {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "elapsed_seconds": round(time.time() - started, 3),
                    }
                )
                print(f"  error at line {line_idx}: {type(exc).__name__}: {exc}", file=sys.stderr)

            append_jsonl(result_jsonl, record)

    print(f"Done. Translated {processed} samples.")
    print(f"Translation log: {result_jsonl}")
    return 0


def main() -> int:
    parser = ConfigArgumentParser(description="Run pre-translation prompt enhancement experiments with Qwen.")
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), required=True)
    parser.add_argument("--prompt-style", choices=["p0", "p1", "p2"], required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input", type=Path, default=None, help="Python directory for avatar/codenet/transcoder, or JSONL for xcodeeval/magicoder/qwen7b/qwen32b.")
    parser.add_argument("--test-dir", type=Path, default=None, help="TransCoder Java test template directory.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-file-size-bytes", type=int, default=5 * 1024)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start-idx", type=int, default=0, help="XCodeEval start line index.")
    parser.add_argument("--end-idx", type=int, default=None, help="XCodeEval end line index.")
    parser.add_argument("--source-field", default="source_code", help="XCodeEval source code field.")
    parser.add_argument("--uid-field", default="src_uid", help="XCodeEval UID field.")
    parser.add_argument('--model-path', type=Path, default=None, help='Local model directory, resolved relative to the repository; overrides --model.')
    parser.add_argument('--model-revision', default=None, help='Exact model revision when known; unset preserves the original loader behavior.')
    parser.add_argument('--prompt-bundle', type=Path, default=None, help='Optional frozen type-hint bundle for prevention experiments.')
    parser.add_argument('--torch-dtype', choices=['auto', 'float16', 'bfloat16', 'float32'], default='auto')
    args = parser.parse_args()
    configure_generation(args)


    if args.greedy:
        args.do_sample = False

    defaults = DATASET_DEFAULTS[args.dataset]
    input_path = args.input or defaults["input"]
    out_dir = (args.out_dir or defaults["out"]) / args.prompt_style
    test_dir = args.test_dir or defaults.get("test")

    print(f"Dataset: {args.dataset}")
    print(f"Prompt style: {args.prompt_style}")
    print(f"Input: {input_path}")
    print(f"Output: {out_dir}")
    print(f"Decoding: {'sampling' if args.do_sample else 'greedy'}")

    model, tokenizer = load_model(args.model)
    kind = defaults["kind"]
    if kind == "dir":
        return translate_dir_dataset(args, args.dataset, input_path, out_dir, model, tokenizer)
    if kind == "jsonl":
        return translate_xcodeeval(args, input_path, out_dir, model, tokenizer)
    if kind == "transcoder":
        return translate_transcoder(args, input_path, test_dir, out_dir, model, tokenizer)
    if kind == "error_jsonl":
        return translate_error_jsonl(args, input_path, out_dir, model, tokenizer)
    raise ValueError(f"Unsupported dataset kind: {kind}")



MODEL_REVISION = None
TORCH_DTYPE = 'auto'
PROMPT_HINTS_OVERRIDE = None
JAVA_CMD, JAVAC_CMD = 'java', 'javac'

def configure_generation(args):
    global DATASET_DEFAULTS, MODEL_REVISION, TORCH_DTYPE, PROMPT_HINTS_OVERRIDE, JAVA_CMD, JAVAC_CMD
    if args.model_path is not None:
        args.model = str(args.model_path)
    DATASET_DEFAULTS = generation_datasets(args.external_data_root)
    MODEL_REVISION = args.model_revision
    TORCH_DTYPE = args.torch_dtype
    JAVA_CMD, JAVAC_CMD = configure_java(args)
    if args.prompt_bundle:
        bundle = json.loads(args.prompt_bundle.read_text(encoding='utf-8-sig'))
        PROMPT_HINTS_OVERRIDE = bundle['type_hints']
        if set(PROMPT_HINTS_OVERRIDE) != {'p0', 'p1', 'p2'}:
            raise ValueError('Prompt bundle must provide p0, p1 and p2')
    datasets = selected_datasets(args.dataset) if hasattr(args, 'model_family') else [args.dataset]
    planned = []
    styles = (['p0', 'p1', 'p2'] if args.prompt_style == 'all' else [args.prompt_style])
    model_name = args.model or resolve_model_path(args.model_family, None)
    tag = getattr(args, 'model_tag', None) or (Path(model_name).name if model_name else 'model')
    base = args.out_dir or REPO_ROOT / 'results/generation' / tag
    for dataset in datasets:
        defaults = DATASET_DEFAULTS[dataset]
        data_input = args.input or defaults['input']
        test_dir = args.test_dir or defaults.get('test')
        if args.input is not None and len(datasets) > 1:
            raise ValueError('--input requires a single dataset')
        for style in styles:
            planned.append({'dataset': dataset, 'prompt_style': style, 'input': data_input,
                            'test_dir': test_dir, 'output': (base / dataset if len(datasets) > 1 else base) / style})
    args.out_dir = base
    if len(datasets) == 1:
        args.input = planned[0]['input']
        args.test_dir = planned[0]['test_dir']
    extra = {'planned_runs': planned, 'model': model_name,
             'prompt_bundle_sha256': __import__('hashlib').sha256(args.prompt_bundle.read_bytes()).hexdigest() if args.prompt_bundle else None,
             'system_prompt_sha256': __import__('hashlib').sha256(args.system_prompt.encode()).hexdigest()}
    finish_setup(args, base / f'run_config_{args.prompt_style}.json', extra)


if __name__ == "__main__":
    raise SystemExit(main())
