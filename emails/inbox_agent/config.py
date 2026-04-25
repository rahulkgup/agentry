from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Monorepo root .env: agentry/.env  <-  emails/inbox_agent/config.py
ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ROOT_ENV)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    user_name: str = os.getenv("USER_NAME", "me")
    user_voice_notes: str = os.getenv(
        "USER_VOICE_NOTES",
        "Casual but professional. Short sentences. No corporate jargon.",
    )

    allowlist_senders: str = os.getenv("ALLOWLIST_SENDERS", "")
    max_emails_per_run: int = int(os.getenv("MAX_EMAILS_PER_RUN", "20"))
    gmail_credentials_path: Path = Path(os.getenv("GMAIL_CREDENTIALS_PATH", "./gmail_credentials.json"))
    gmail_token_path: Path = Path(os.getenv("GMAIL_TOKEN_PATH", "./gmail_token.json"))

    @property
    def allowlist(self) -> set[str]:
        if not self.allowlist_senders.strip():
            return set()
        return {s.strip().lower() for s in self.allowlist_senders.split(",") if s.strip()}

    @property
    def fake_llm(self) -> bool:
        # Read each time so Typer/CLI can set FAKE_LLM=1 in-process.
        return os.getenv("FAKE_LLM", "").lower() in ("1", "true", "yes")


settings = Settings()
