"""Shared, offline configuration and path handling for repository entry points."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PATH_ARGUMENTS = {
    "input", "out", "out_dir", "java_dir", "test_dir", "test_case_dir",
    "filled_dir", "raw_dir", "summary_out", "unittest_db", "claude_file",
    "gemini_file", "external_data_root", "jdk_bin", "prevention_results",
    "prevention_generation", "repair_results", "repair_generation", "prompt_bundle", "model_path",
}


def repo_path(value) -> Path:
    value = os.path.expandvars(str(value))
    if re.search(r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z_0-9]*|%[A-Za-z_][A-Za-z_0-9]*%", value):
        raise ValueError(f"Unset environment variable in path: {value}")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def external_root() -> Path:
    return repo_path(os.environ.get("TYPE_ERROR_DATA_ROOT") or "external_data")


def generation_datasets(data_root: Path | None = None) -> dict:
    data_root = data_root or external_root()
    result = {
        "avatar": {"kind": "dir", "input": data_root / "avatar/Code"},
        "codenet": {"kind": "dir", "input": data_root / "codenet/Code"},
        "xcodeeval": {"kind": "jsonl", "input": data_root / "xcodeeval/Python.jsonl"},
        "transcoder": {"kind": "transcoder", "input": data_root / "transcoder/python/Code", "test": data_root / "transcoder/java"},
    }
    for model in ("magicoder", "qwen7b", "qwen32b"):
        result[model] = {"kind": "error_jsonl", "input": REPO_ROOT / "dataset/type_error_dataset.jsonl"}
    for name, item in result.items():
        item["out"] = REPO_ROOT / "results/generation" / name
    return result


def matches_error_source(sample: dict, selected_model: str) -> bool:
    """Select cases by their original translation model from the shared corpus."""
    names = {
        "magicoder": "Magicoder-S-DS-6.7B",
        "qwen7b": "Qwen7B",
        "qwen32b": "Qwen32B",
    }
    value = sample.get("model")
    if value is None:
        raise ValueError("Type-error input records must include the original translation model")
    aliases = {key.casefold(): key for key in names}
    aliases.update({value.casefold(): key for key, value in names.items()})
    source = aliases.get(str(value).strip().casefold())
    if source is None:
        raise ValueError(f"Unknown original translation model: {value!r}")
    return source == selected_model


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items() if k not in {"api_key", "API_KEY", "headers"}}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return value


class ConfigArgumentParser(argparse.ArgumentParser):
    """JSON args supply defaults; explicit CLI flags win. Paths are repo-relative."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--config", type=Path, help="JSON experiment config; relative to repository root.")
        self.add_argument("--dry-run", action="store_true", help="Print effective config and stop without API/model/test execution or output writes.")
        self.add_argument("--external-data-root", type=Path, default=external_root(), help="External resources root; TYPE_ERROR_DATA_ROOT or external_data by default.")
        self.add_argument("--jdk-bin", type=Path, default=None, help="JDK bin directory; JAVA_HOME/bin or PATH when omitted.")
        self.add_argument("--seed", type=int, default=None, help="Optional local random seed. Historical unspecified seeds remain unset.")

    def parse_args(self, args=None, namespace=None):
        argv = list(sys.argv[1:] if args is None else args)
        probe = argparse.ArgumentParser(add_help=False)
        probe.add_argument("--config", type=Path)
        preliminary, _ = probe.parse_known_args(argv)
        document = {}
        config_path = None
        actions = {a.dest: a for a in self._actions}
        if preliminary.config:
            try:
                config_path = repo_path(preliminary.config)
                document = json.loads(config_path.read_text(encoding="utf-8-sig"))
                if not isinstance(document, dict) or not isinstance(document.get("args", {}), dict):
                    raise ValueError("Config must be an object with an args object")
                overrides = document.get("args", {})
                unknown = set(overrides) - set(actions)
                if unknown:
                    raise ValueError(f"Unknown config arguments: {sorted(unknown)}")
                if set(overrides) & {"config", "dry_run", "help"}:
                    raise ValueError("config/dry_run/help belong on the CLI")
                for key, value in overrides.items():
                    action = actions[key]
                    if value is not None:
                        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction)):
                            if not isinstance(value, bool):
                                raise ValueError(f"{key} must be a JSON boolean")
                        elif action.type is not None:
                            if action.nargs in {"+", "*"} or isinstance(action.nargs, int):
                                if not isinstance(value, list):
                                    raise ValueError(f"{key} must be a list")
                                value = [action.type(v) for v in value]
                            else:
                                if isinstance(value, bool):
                                    raise ValueError(f"{key} has the wrong type")
                                value = action.type(value)
                        if action.choices is not None:
                            choices = value if isinstance(value, list) else [value]
                            if any(v not in action.choices for v in choices):
                                raise ValueError(f"Invalid {key}: {value}; choices={list(action.choices)}")
                    action.default = value
                    if value is not None:
                        action.required = False
                        if not action.option_strings and action.nargs == "+":
                            action.nargs = "*"
            except (OSError, ValueError, TypeError) as exc:
                self.error(str(exc))
        result = super().parse_args(argv, namespace)
        try:
            for name in PATH_ARGUMENTS:
                value = getattr(result, name, None)
                if value is not None:
                    setattr(result, name, repo_path(value))
            if hasattr(result, "files"):
                result.files = [repo_path(v) for v in result.files]
        except ValueError as exc:
            self.error(str(exc))
        if getattr(result, "greedy", False):
            result.do_sample = False
        for negative, positive in (("no_javac", "run_javac"), ("no_tests", "run_tests")):
            if getattr(result, negative, False):
                setattr(result, positive, False)
        result._experiment = document.get("experiment", {})
        result._config_path = config_path
        result._script = Path(sys.argv[0]).resolve()
        return result


def configure_java(args) -> tuple[str, str]:
    jdk_bin = args.jdk_bin
    if jdk_bin is None and os.environ.get("JAVA_HOME"):
        jdk_bin = Path(os.environ["JAVA_HOME"]).expanduser() / "bin"
    suffix = ".exe" if os.name == "nt" else ""
    if jdk_bin:
        for command in ("java", "javac"):
            path = jdk_bin / (command + suffix)
            if not path.is_file() and not args.dry_run:
                raise ValueError(f"JDK executable missing: {path}")
        return str(jdk_bin / ("java" + suffix)), str(jdk_bin / ("javac" + suffix))
    return "java", "javac"


def finish_setup(args, output: Path | None = None, extra: dict | None = None):
    """Record effective settings before work. Does not write anything in dry runs."""
    settings = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    # A provider URL may include a credential query. Never record that query.
    if settings.get("base_url"):
        settings["base_url"] = settings["base_url"].split("?")[0]
    record = {
        "experiment": args._experiment,
        "args": json_safe(settings),
        "resolved": json_safe(extra or {}),
        "repository_root": str(REPO_ROOT),
        "config": str(args._config_path) if args._config_path else None,
        "script_sha256": hashlib.sha256(args._script.read_bytes()).hexdigest(),
    }
    if args.dry_run:
        print(json.dumps(record, ensure_ascii=False, indent=2))
        raise SystemExit(0)
    resources = []
    inputs = [(name, getattr(args, name, None)) for name in
              ("input", "claude_file", "gemini_file", "unittest_db")]
    inputs.extend(("files", path) for path in getattr(args, "files", []))
    for role, path in inputs:
        if isinstance(path, Path) and path.is_file():
            checksum = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    checksum.update(chunk)
            resources.append({"role": role, "path": str(path), "bytes": path.stat().st_size,
                              "sha256": checksum.hexdigest()})
    record["input_resources"] = resources
    if output is not None and getattr(args, "resume", False) and Path(output).is_file():
        previous = json.loads(Path(output).read_text(encoding="utf-8-sig"))
        nonsemantic = {"config", "dry_run", "resume", "limit", "print_usage",
                       "request_timeout", "max_retries", "sleep_seconds"}
        old_args = {k: v for k, v in previous.get("args", {}).items() if k not in nonsemantic}
        new_args = {k: v for k, v in record["args"].items() if k not in nonsemantic}
        if (old_args != new_args or previous.get("script_sha256") != record["script_sha256"]
                or previous.get("input_resources") != record["input_resources"]):
            raise ValueError("Resume settings, script or input content differ from the saved run; choose a new output location")
    if args.seed is not None:
        random.seed(args.seed)
        # Set before importing CUDA/model libraries. Seed torch lazily in model loaders.
        os.environ["TYPE_ERROR_RUN_SEED"] = str(args.seed)
    else:
        os.environ.pop("TYPE_ERROR_RUN_SEED", None)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        record["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def seed_torch():
    value = os.environ.get("TYPE_ERROR_RUN_SEED")
    if value is not None:
        import torch
        torch.manual_seed(int(value))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(value))


def model_dtype(name: str):
    if name == "auto":
        return name
    import torch
    return getattr(torch, name)


def credential(env_name: str) -> str:
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise ValueError(f"Set the {env_name} environment variable before making API requests")
    return value
