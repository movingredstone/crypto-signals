"""Allowlisted command runner.

SECURITY: this NEVER runs free-form text. It only runs commands that the user
pre-registered under `commands:` in config.yaml, looked up by name. DeepSeek (or
any chat input) can only choose a registered NAME; the actual command string is
the fixed one from config. No arbitrary shell, no command injection surface.
"""
import os
import json
import signal
import subprocess
from pathlib import Path
from datetime import datetime, timezone

from src.config_loader import load_config


STATE_PATH = Path("logs/running_commands.json")


def get_commands(config_path: str = "config.yaml") -> dict:
    cfg = load_config(config_path)
    cmds = cfg.get("commands") or {}
    return cmds if isinstance(cmds, dict) else {}


def _load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _expand(p: str) -> str:
    return str(Path(os.path.expanduser(str(p))))


def _alive(pid) -> bool:
    try:
        pid = int(pid)
    except Exception:
        return False
    # If it's our own child that already exited, reap it (avoid zombie misreport).
    try:
        wpid, _ = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            return False
    except ChildProcessError:
        pass  # not our child (e.g. survived a bot restart) -> fall through
    except Exception:
        pass
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def list_commands(config_path: str = "config.yaml") -> list[dict]:
    cmds = get_commands(config_path)
    state = _load_state()
    out = []
    for name, spec in cmds.items():
        st = state.get(name)
        running = bool(st and _alive(st.get("pid", -1)))
        out.append({
            "name": name,
            "desc": (spec or {}).get("desc", ""),
            "running": running,
            "pid": st.get("pid") if st else None,
            "log": st.get("log") if st else None,
        })
    return out


def start_command(name: str, config_path: str = "config.yaml") -> dict:
    cmds = get_commands(config_path)
    if name not in cmds:
        return {"ok": False, "error": f"등록되지 않은 명령: {name}"}

    spec = cmds[name] or {}
    run = spec.get("run")
    if not run:
        return {"ok": False, "error": f"'{name}'에 run 정의가 없음"}

    cwd = _expand(spec.get("cwd", "~"))
    if not Path(cwd).is_dir():
        return {"ok": False, "error": f"작업 디렉토리가 없음: {cwd}"}

    background = bool(spec.get("background", True))

    state = _load_state()
    existing = state.get(name)
    if existing and _alive(existing.get("pid", -1)):
        return {
            "ok": False,
            "error": f"이미 실행 중 (pid {existing['pid']})",
            "pid": existing["pid"],
            "log": existing.get("log"),
        }

    log_path = Path("logs") / f"cmd_{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "a", encoding="utf-8")
    logf.write(f"\n=== start {datetime.now(timezone.utc).isoformat()} :: {run} ===\n")
    logf.flush()

    # Login shell so `source ... && npm ...` works. Own session => killable as a group.
    proc = subprocess.Popen(
        ["bash", "-lc", run],
        cwd=cwd,
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    if not background:
        timeout = int(spec.get("timeout", 120))
        try:
            proc.wait(timeout=timeout)
            return {"ok": True, "background": False, "pid": proc.pid, "log": str(log_path),
                    "run": run, "cwd": cwd, "exit_code": proc.returncode}
        except subprocess.TimeoutExpired:
            return {"ok": True, "background": False, "pid": proc.pid, "log": str(log_path),
                    "run": run, "cwd": cwd, "note": f"{timeout}s 안에 안 끝남(계속 실행 중)"}

    state[name] = {
        "pid": proc.pid,
        "started": datetime.now(timezone.utc).isoformat(),
        "log": str(log_path),
        "run": run,
        "cwd": cwd,
    }
    _save_state(state)
    return {"ok": True, "background": True, "pid": proc.pid, "log": str(log_path),
            "run": run, "cwd": cwd}


def stop_command(name: str, config_path: str = "config.yaml") -> dict:
    state = _load_state()
    st = state.get(name)
    if not st or not _alive(st.get("pid", -1)):
        # clean stale entry
        if name in state:
            del state[name]
            _save_state(state)
        return {"ok": False, "error": f"'{name}'는 실행 중이 아님"}

    pid = int(st["pid"])
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception as e:
            return {"ok": False, "error": f"종료 실패: {type(e).__name__}: {e}"}

    del state[name]
    _save_state(state)
    return {"ok": True, "pid": pid, "name": name}
