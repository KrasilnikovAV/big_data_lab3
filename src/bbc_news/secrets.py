from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from ansible.parsing.vault import VaultLib, VaultSecret


DEFAULT_VAULT_IDENTITY = "default"
DEFAULT_VAULT_FILE_CANDIDATES = (
    Path("/app/secrets/clickhouse.vault.yml"),
    Path("secrets/clickhouse.vault.yml"),
)
DEFAULT_VAULT_PASSWORD_FILE_CANDIDATES = (
    Path("/run/secrets/.vault_pass.txt"),
    Path("/run/secrets/ansible_vault_password"),
    Path("secrets/.vault_pass.txt"),
)


def load_ansible_vault(
    vault_file: str | Path | None = None,
    password_file: str | Path | None = None,
) -> dict[str, Any]:
    resolved_vault_file = _resolve_vault_file(vault_file)
    resolved_password = _resolve_password(password_file=password_file)
    cipher_text = resolved_vault_file.read_bytes()
    vault = VaultLib([(DEFAULT_VAULT_IDENTITY, VaultSecret(resolved_password.encode("utf-8")))])
    plain_text = vault.decrypt(cipher_text).decode("utf-8")

    payload = yaml.safe_load(plain_text) or {}
    if not isinstance(payload, dict):
        raise ValueError("Ansible Vault payload must be a mapping.")
    return payload


def load_secret_value(
    name: str,
    default: str | None = None,
    vault_file: str | Path | None = None,
    password_file: str | Path | None = None,
) -> str | None:
    vault_payload = load_ansible_vault(vault_file=vault_file, password_file=password_file)
    value = vault_payload.get(name, default)
    if value is None:
        return None
    return str(value).strip()


def _resolve_vault_file(vault_file: str | Path | None) -> Path:
    if vault_file is not None:
        resolved = Path(vault_file).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Ansible Vault file was not found: {resolved}")
        return resolved

    for candidate in DEFAULT_VAULT_FILE_CANDIDATES:
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        "Ansible Vault file was not found. Checked: "
        + ", ".join(str(path) for path in DEFAULT_VAULT_FILE_CANDIDATES)
    )


def _resolve_password(password_file: str | Path | None) -> str:
    candidates = [Path(password_file).expanduser()] if password_file is not None else list(
        DEFAULT_VAULT_PASSWORD_FILE_CANDIDATES
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.exists():
            continue
        file_password = resolved.read_text(encoding="utf-8").strip()
        if file_password:
            return file_password

    raise FileNotFoundError(
        "Ansible Vault password file was not found or is empty. Checked: "
        + ", ".join(str(path) for path in candidates)
    )
