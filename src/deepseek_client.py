import os
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv


RAW_LOG_PATH = Path("logs/deepseek_router_raw.log")


def log_raw_response(tag: str, raw: str) -> None:
    """Persist the raw DeepSeek response so invalid-JSON cases are inspectable."""
    snippet = (raw or "")[:4000]
    try:
        RAW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RAW_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"\n--- {datetime.now(timezone.utc).isoformat()} {tag} ---\n{snippet}\n")
    except Exception:
        pass
    print(f"[deepseek] raw[{tag}]: {snippet[:300]!r}", flush=True)


def try_extract_json(raw: str):
    """Best-effort JSON extraction. Returns dict on success, None on failure."""
    try:
        return extract_json(raw)
    except Exception:
        return None


DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

KEY_NAMES = [
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_KEY",
    "DEEPSEEK_TOKEN",
    "DEEPSEEK_API_TOKEN",
    "DEEPSEEK_SECRET",
]

MODEL_NAMES = [
    "DEEPSEEK_MODEL",
]

SKIP_DIRS = {
    ".venv", "venv", "node_modules", ".git", "__pycache__",
    "Library", "Applications", "Movies", "Music", "Pictures",
    ".Trash", "Caches", "Containers", "Group Containers",
}

SEARCH_EXTS = {
    ".env", ".txt", ".json", ".yaml", ".yml", ".toml",
    ".zshrc", ".bashrc", ".profile", ".bash_profile", ".cfg", ".conf"
}


def _strip_value(v: str) -> str:
    v = v.strip()
    if v.startswith("export "):
        v = v.replace("export ", "", 1).strip()
    v = v.strip().strip('"').strip("'").strip()
    return v


def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _parse_key_value_text(text: str) -> dict:
    found = {}

    for line in text.splitlines():
        raw = line.strip()

        if not raw or raw.startswith("#"):
            continue

        if raw.startswith("export "):
            raw = raw.replace("export ", "", 1).strip()

        if "=" in raw:
            k, v = raw.split("=", 1)
            found[k.strip()] = _strip_value(v)

        # JSON/YAML 스타일도 대충 인식
        m = re.match(r'["\']?([A-Za-z0-9_]*DEEPSEEK[A-Za-z0-9_]*)["\']?\s*[:=]\s*["\']?([^"\',\s]+)', raw)
        if m:
            found[m.group(1).strip()] = _strip_value(m.group(2))

    return found


def _candidate_files() -> list[Path]:
    cwd = Path.cwd()
    home = Path.home()

    return [
        cwd / ".env",
        cwd.parent / ".env",
        home / "Desktop" / "hermes" / ".env",
        home / "Desktop" / "hermes" / "investmentsystem" / ".env",
        home / ".env",
        home / ".zshrc",
        home / ".bashrc",
        home / ".bash_profile",
        home / ".profile",
        home / ".config" / "deepseek" / ".env",
        home / ".config" / "deepseek.json",
        home / ".config" / "openai" / ".env",
    ]


def _find_key_in_text_file(path: Path):
    text = _read_file(path)
    if not text:
        return None

    data = _parse_key_value_text(text)

    for name in KEY_NAMES:
        if data.get(name):
            model = data.get("DEEPSEEK_MODEL") or "deepseek-chat"
            return data[name], model, str(path)

    # 변수명이 다르더라도 DEEPSEEK가 들어간 키를 자동 인식
    for k, v in data.items():
        if "DEEPSEEK" in k.upper() and v and len(v) > 10:
            model = data.get("DEEPSEEK_MODEL") or "deepseek-chat"
            return v, model, str(path)

    # 텍스트 안에 DeepSeek key로 보이는 패턴이 있는 경우
    # 값은 출력하지 않고 내부에서만 사용
    if "deepseek" in text.lower():
        m = re.search(r'(sk-[A-Za-z0-9_\-]{20,})', text)
        if m:
            return m.group(1), "deepseek-chat", str(path)

    return None


def _search_home_for_deepseek_key():
    home = Path.home()

    roots = [
        home / "Desktop",
        home / "Documents",
        home / ".config",
        home,
    ]

    checked = 0
    max_files = 3000

    for root in roots:
        if not root.exists():
            continue

        # home 자체에서는 깊게 들어가지 않도록 주요 파일만 먼저 처리
        if root == home:
            for path in [home / ".zshrc", home / ".bashrc", home / ".profile", home / ".bash_profile", home / ".env"]:
                result = _find_key_in_text_file(path)
                if result:
                    return result
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".Trash")]

            for filename in filenames:
                checked += 1
                if checked > max_files:
                    return None

                lower = filename.lower()
                path = Path(dirpath) / filename

                should_check = (
                    "env" in lower
                    or "secret" in lower
                    or "key" in lower
                    or "deepseek" in lower
                    or path.suffix.lower() in SEARCH_EXTS
                )

                if not should_check:
                    continue

                result = _find_key_in_text_file(path)
                if result:
                    return result

    return None


def resolve_deepseek_config() -> tuple[str, str, str]:
    # 1) 현재 shell 환경변수
    for name in KEY_NAMES:
        val = os.getenv(name)
        if val:
            model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            return val.strip(), model.strip(), f"environment variable: {name}"

    # 2) 기본 dotenv
    load_dotenv(override=False)

    for name in KEY_NAMES:
        val = os.getenv(name)
        if val:
            model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            return val.strip(), model.strip(), f"loaded by python-dotenv: {name}"

    # 3) 흔한 후보 파일
    for file_path in _candidate_files():
        result = _find_key_in_text_file(file_path)
        if result:
            return result

    # 4) 홈 폴더 제한 자동 검색
    result = _search_home_for_deepseek_key()
    if result:
        return result

    searched = "\n".join(f"- {p}" for p in _candidate_files())

    raise ValueError(
        "DeepSeek API key를 자동으로 찾지 못했습니다.\n\n"
        "자동 검색 위치:\n"
        f"{searched}\n"
        "- ~/Desktop\n"
        "- ~/Documents\n"
        "- ~/.config\n\n"
        "키가 있는 파일에 DEEPSEEK_API_KEY=... 또는 DEEPSEEK_KEY=... 형태로 저장되어 있어야 합니다."
    )


def extract_json(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("DeepSeek response did not contain valid JSON.")

    return json.loads(match.group(0))


def call_deepseek_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 4000,
) -> dict:
    api_key, model, source = resolve_deepseek_config()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    r = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=180)

    if r.status_code >= 400:
        raise RuntimeError(
            f"DeepSeek API error {r.status_code}: {r.text[:1000]}\n"
            f"Key source: {source}\n"
            f"Model: {model}"
        )

    data = r.json()
    content = data["choices"][0]["message"]["content"]

    return extract_json(content)



def call_deepseek_text(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 1800,
) -> str:
    api_key, model, source = resolve_deepseek_config()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    r = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=180)

    if r.status_code >= 400:
        raise RuntimeError(
            f"DeepSeek API error {r.status_code}: {r.text[:1000]}\n"
            f"Key source: {source}\n"
            f"Model: {model}"
        )

    data = r.json()
    return data["choices"][0]["message"]["content"].strip()
