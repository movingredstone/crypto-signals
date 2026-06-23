#!/usr/bin/env python3
"""Safely switch the active bmu-android Hermes profile to a new Telegram bot.

This script prompts for a BotFather token without echoing it, updates
~/.hermes/profiles/bmu-android/.env, and creates a timestamped backup first.
"""
from __future__ import annotations

import getpass
import re
import shutil
from datetime import datetime
from pathlib import Path

ENV_PATH = Path.home() / ".hermes" / "profiles" / "bmu-android" / ".env"
TOKEN_RE = re.compile(r"^\d{6,}:[A-Za-z0-9_-]{20,}$")


def parse_env(text: str) -> list[tuple[str | None, str]]:
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            rows.append((None, line))
            continue
        key, value = line.split("=", 1)
        rows.append((key, value))
    return rows


def set_key(rows: list[tuple[str | None, str]], key: str, value: str) -> None:
    for i, (k, _v) in enumerate(rows):
        if k == key:
            rows[i] = (key, value)
            return
    rows.append((key, value))


def render(rows: list[tuple[str | None, str]]) -> str:
    out = []
    for key, value in rows:
        if key is None:
            out.append(value)
        else:
            out.append(f"{key}={value}")
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    if not ENV_PATH.exists():
        raise SystemExit(f"Missing env file: {ENV_PATH}")

    print(f"Hermes env: {ENV_PATH}")
    token = getpass.getpass("Paste new BotFather token (hidden): ").strip()
    if not TOKEN_RE.match(token):
        raise SystemExit("Token format does not look like a Telegram bot token. Aborted.")

    clear_allowed = input(
        "Clear TELEGRAM_ALLOWED_USERS so the new Telegram account can connect? [y/N] "
    ).strip().lower() == "y"
    clear_home = input(
        "Clear TELEGRAM_HOME_CHANNEL so you can set the new chat with /sethome? [Y/n] "
    ).strip().lower() != "n"

    backup = ENV_PATH.with_suffix(ENV_PATH.suffix + f".bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(ENV_PATH, backup)

    rows = parse_env(ENV_PATH.read_text())
    set_key(rows, "TELEGRAM_BOT_TOKEN", token)
    if clear_allowed:
        set_key(rows, "TELEGRAM_ALLOWED_USERS", "")
    if clear_home:
        set_key(rows, "TELEGRAM_HOME_CHANNEL", "")

    ENV_PATH.write_text(render(rows))
    print(f"Updated {ENV_PATH}")
    print(f"Backup: {backup}")
    print("Next: hermes --profile bmu-android gateway restart")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
