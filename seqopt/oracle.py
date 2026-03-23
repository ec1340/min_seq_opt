"""Client for the ligand design challenge oracle API."""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests  # type: ignore[reportMissingModuleSource]


@dataclass
class OracleClient:
    """HTTP client wrapper for register/query/submit endpoints."""

    base_url: str | None = None

    def __post_init__(self) -> None:
        if self.base_url is None:
            self.base_url = os.getenv("SEQOPT_ORACLE_BASE_URL")
        if not self.base_url:
            raise ValueError(
                "Oracle base URL is not configured. Pass base_url=... or set SEQOPT_ORACLE_BASE_URL."
            )

    def register(self, email: str) -> dict:
        response = requests.post(f"{self.base_url}/register", json={"email": email}, timeout=30)
        response.raise_for_status()
        return response.json()

    def info(self) -> dict:
        response = requests.get(f"{self.base_url}/info", timeout=30)
        response.raise_for_status()
        return response.json()

    def query(self, token: str, ligand_sequence: str, target_type: str = "target") -> dict:
        response = requests.post(
            f"{self.base_url}/oracle",
            json={"token": token, "ligand": ligand_sequence, "target_type": target_type},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def submit(self, token: str, ligand_sequence: str) -> dict:
        response = requests.post(
            f"{self.base_url}/submit",
            json={"token": token, "ligand": ligand_sequence},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

