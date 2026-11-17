import atexit
import csv
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path


ACTIVE_PROCESSES: set[subprocess.Popen] = set()
WATCHED_IMAGES = {"java.exe", "javac.exe"}
_CLEANING_UP = False


def _tool_path(env_name: str, exe_name: str) -> str:
    if os.name != 'nt' and exe_name.endswith('.exe'):
        exe_name = exe_name[:-4]
    explicit = os.environ.get(env_name)
    if explicit and Path(explicit).exists():
        return explicit
    java_home = os.environ.get('JAVA_HOME')
    if java_home:
        candidate = Path(java_home) / 'bin' / exe_name
        if candidate.exists():
            return str(candidate)
    return shutil.which(exe_name) or exe_name


JAVA_CMD = _tool_path("JAVA_BIN", "java.exe" if os.name == "nt" else "java")
JAVAC_CMD = _tool_path("JAVAC_BIN", "javac.exe" if os.name == "nt" else "javac")


def snapshot_watched_pids() -> set[int]:
    if os.name != "nt":
        return set()
    try:
        result = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            pids: set[int] = set()
            for row in csv.reader(result.stdout.splitlines()):
                if len(row) < 2:
                    continue
                if row[0].strip('"').lower() in WATCHED_IMAGES:
                    try:
                        pid = int(row[1].strip('"'))
                    except ValueError:
                        continue
                    if pid != os.getpid():
                        pids.add(pid)
            return pids
    except Exception:
        pass

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-Process java,javac -ErrorAction SilentlyContinue | "
                "Select-Object -ExpandProperty Id",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        pids: set[int] = set()
        for line in result.stdout.splitlines():
            try:
                pid = int(line.strip())
            except ValueError:
                continue
            if pid != os.getpid():
                pids.add(pid)
        return pids
    except Exception:
        return set()


BASELINE_WATCHED_PIDS = snapshot_watched_pids()


def kill_pid_tree(pid: int) -> None:
    if pid == os.getpid():
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            return
        except Exception:
            pass
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


def kill_new_watched_processes(before_pids: set[int]) -> None:
    try:
        for _ in range(3):
            current = snapshot_watched_pids()
            new_pids = current - before_pids - {os.getpid()}
            if not new_pids:
                return
            for pid in sorted(new_pids):
                kill_pid_tree(pid)
            time.sleep(0.5)
    except BaseException:
        if not _CLEANING_UP:
            raise


def popen_flags() -> int:
    if os.name != "nt":
        return 0
    return subprocess.CREATE_NEW_PROCESS_GROUP


def kill_process_tree(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except Exception:
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass
    try:
        process.kill()
    except Exception:
        pass
    try:
        process.wait(timeout=3)
    except Exception:
        pass


def kill_active_processes() -> None:
    global _CLEANING_UP
    if _CLEANING_UP:
        return
    _CLEANING_UP = True
    try:
        for process in list(ACTIVE_PROCESSES):
            kill_process_tree(process)
        ACTIVE_PROCESSES.clear()
        kill_new_watched_processes(BASELINE_WATCHED_PIDS)
    finally:
        _CLEANING_UP = False


def _signal_cleanup(signum, frame) -> None:
    if _CLEANING_UP:
        return
    kill_active_processes()
    raise SystemExit(128 + signum)


for _sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
    if _sig is not None:
        try:
            signal.signal(_sig, _signal_cleanup)
        except Exception:
            pass


atexit.register(kill_active_processes)


def run_process(
    args: list[str],
    cwd: Path,
    timeout: int,
    input_data: str | bytes | None = None,
    text: bool = True,
) -> tuple[str, subprocess.CompletedProcess | None, str]:
    process = None
    before_pids = snapshot_watched_pids()
    try:
        process = subprocess.Popen(
            args,
            cwd=cwd,
            stdin=subprocess.PIPE if input_data is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=text,
            creationflags=popen_flags(),
            start_new_session=(os.name != "nt"),
        )
        ACTIVE_PROCESSES.add(process)
        stdout, stderr = process.communicate(input=input_data, timeout=timeout)
        return (
            "ok",
            subprocess.CompletedProcess(args, process.returncode, stdout, stderr),
            "",
        )
    except subprocess.TimeoutExpired as exc:
        kill_process_tree(process)
        stdout = exc.stdout or ("" if text else b"")
        stderr = exc.stderr or ("" if text else b"")
        return "timeout", None, f"Timeout after {timeout}s\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    except Exception as exc:
        kill_process_tree(process)
        return "launch_error", None, str(exc)
    finally:
        if process is not None:
            ACTIVE_PROCESSES.discard(process)
        kill_new_watched_processes(before_pids)
