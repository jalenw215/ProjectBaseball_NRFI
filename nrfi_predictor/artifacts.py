from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_BACKTEST_FILE,
    DEFAULT_FETCH_STATE_FILE,
    DEFAULT_MODEL_FILE,
    DEFAULT_PREDICTIONS_FILE,
    DEFAULT_REFRESH_LOG,
    DEFAULT_STATCAST_FILE,
    DEFAULT_TRAINING_FILE,
)
from .utils import ensure_parent

APP_NAME = "nrfi"
DEFAULT_BUCKET = "baseball-artifacts"


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_type: str
    local_path: Path
    relative_path: str
    content_type: str


ARTIFACT_SPECS = {
    "statcast": ArtifactSpec("statcast", DEFAULT_STATCAST_FILE, "raw/statcast_history.csv", "text/csv"),
    "training": ArtifactSpec("training", DEFAULT_TRAINING_FILE, "processed/training_rows.csv", "text/csv"),
    "model": ArtifactSpec("model", DEFAULT_MODEL_FILE, "models/nrfi_model.joblib", "application/octet-stream"),
    "predictions": ArtifactSpec("predictions", DEFAULT_PREDICTIONS_FILE, "predictions/latest_predictions.csv", "text/csv"),
    "backtest": ArtifactSpec("backtest", DEFAULT_BACKTEST_FILE, "processed/backtest_predictions.csv", "text/csv"),
    "fetch_state": ArtifactSpec("fetch_state", DEFAULT_FETCH_STATE_FILE, "logs/historical_fetch_state.json", "application/json"),
    "refresh_log": ArtifactSpec("refresh_log", DEFAULT_REFRESH_LOG, "logs/refresh.log", "text/plain"),
}


class SupabaseArtifacts:
    def __init__(self, app_name: str = APP_NAME):
        self.app_name = app_name
        self.url = _secret("SUPABASE_URL")
        self.service_role_key = _secret("SUPABASE_SERVICE_ROLE_KEY")
        self.bucket = _secret("SUPABASE_ARTIFACT_BUCKET") or DEFAULT_BUCKET
        self._client = None

    @property
    def configured(self) -> bool:
        return bool(self.url and self.service_role_key)

    def storage_path(self, artifact_type: str) -> str:
        spec = artifact_spec(artifact_type)
        return f"{self.app_name}/{spec.relative_path}"

    def local_path(self, artifact_type: str) -> Path:
        return artifact_spec(artifact_type).local_path

    def download_if_missing(self, artifact_type: str, local_path: Path | None = None) -> bool:
        local_path = local_path or self.local_path(artifact_type)
        if local_path.exists():
            return True
        return self.download(artifact_type, local_path)

    def download(self, artifact_type: str, local_path: Path | None = None) -> bool:
        if not self.configured:
            return False
        spec = artifact_spec(artifact_type)
        local_path = local_path or spec.local_path
        storage_path = self.latest_storage_path(artifact_type) or self.storage_path(artifact_type)
        try:
            payload = self.client.storage.from_(self.bucket).download(storage_path)
            ensure_parent(local_path).write_bytes(_as_bytes(payload))
            return True
        except Exception:
            return False

    def upload(self, artifact_type: str, local_path: Path | None = None, metadata: dict[str, Any] | None = None) -> bool:
        if not self.configured:
            return False
        spec = artifact_spec(artifact_type)
        local_path = local_path or spec.local_path
        if not local_path.exists():
            return False
        storage_path = self.storage_path(artifact_type)
        data = local_path.read_bytes()
        options = {"content-type": spec.content_type, "upsert": "true"}
        try:
            bucket = self.client.storage.from_(self.bucket)
            try:
                bucket.upload(storage_path, data, file_options=options)
            except Exception:
                bucket.update(storage_path, data, file_options=options)
            self._record_manifest(spec, local_path, storage_path, metadata or {})
            return True
        except Exception:
            return False

    def artifact_ready(self, artifact_type: str, local_path: Path | None = None) -> bool:
        local_path = local_path or self.local_path(artifact_type)
        if local_path.exists():
            return True
        if not self.configured:
            return False
        return self.latest_storage_path(artifact_type) is not None

    def latest_storage_path(self, artifact_type: str) -> str | None:
        if not self.configured:
            return None
        try:
            result = (
                self.client.table("artifact_manifest")
                .select("storage_path")
                .eq("app_name", self.app_name)
                .eq("artifact_type", artifact_type)
                .eq("is_latest", True)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            data = result.data or []
            return data[0]["storage_path"] if data else None
        except Exception:
            return None

    def read_refresh_state(self) -> dict[str, Any]:
        if not self.configured:
            return {}
        try:
            result = self.client.table("refresh_state").select("*").eq("app_name", self.app_name).limit(1).execute()
            data = result.data or []
            return data[0] if data else {}
        except Exception:
            return {}

    def upsert_refresh_state(self, values: dict[str, Any]) -> bool:
        if not self.configured:
            return False
        payload = {"app_name": self.app_name, "updated_at": datetime.now(timezone.utc).isoformat(), **values}
        try:
            self.client.table("refresh_state").upsert(payload, on_conflict="app_name").execute()
            return True
        except Exception:
            return False

    def log_pipeline_run(
        self,
        job_name: str,
        status: str,
        detail: str = "",
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> bool:
        if not self.configured:
            return False
        payload = {
            "app_name": self.app_name,
            "job_name": job_name,
            "status": status,
            "started_at": (started_at or datetime.now(timezone.utc)).isoformat(),
            "finished_at": (finished_at or datetime.now(timezone.utc)).isoformat(),
            "detail": detail,
            "error": error,
            "metadata": metadata or {},
        }
        try:
            self.client.table("pipeline_runs").insert(payload).execute()
            return True
        except Exception:
            return False

    @property
    def client(self):
        if self._client is None:
            from supabase import create_client

            self._client = create_client(str(self.url), str(self.service_role_key))
        return self._client

    def _record_manifest(
        self,
        spec: ArtifactSpec,
        local_path: Path,
        storage_path: str,
        metadata: dict[str, Any],
    ) -> None:
        self.client.table("artifact_manifest").update({"is_latest": False}).eq("app_name", self.app_name).eq(
            "artifact_type", spec.artifact_type
        ).execute()
        payload = {
            "app_name": self.app_name,
            "artifact_type": spec.artifact_type,
            "storage_path": storage_path,
            "content_type": spec.content_type,
            "size_bytes": local_path.stat().st_size,
            "sha256": hashlib.sha256(local_path.read_bytes()).hexdigest(),
            "is_latest": True,
            "metadata": metadata,
        }
        self.client.table("artifact_manifest").insert(payload).execute()


def artifact_spec(artifact_type: str) -> ArtifactSpec:
    try:
        return ARTIFACT_SPECS[artifact_type]
    except KeyError as exc:
        raise ValueError(f"Unknown artifact type: {artifact_type}") from exc


def _secret(name: str) -> str | None:
    if os.getenv(name):
        return os.getenv(name)
    try:
        import streamlit as st

        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:
        return None


def _as_bytes(payload) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if isinstance(payload, str):
        return payload.encode("utf-8")
    return bytes(payload)
