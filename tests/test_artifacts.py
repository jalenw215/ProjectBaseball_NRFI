from __future__ import annotations

from pathlib import Path

from nrfi_predictor.artifacts import SupabaseArtifacts, artifact_spec


def test_artifact_paths_are_namespaced():
    store = SupabaseArtifacts(app_name="nrfi")

    assert store.storage_path("statcast") == "nrfi/raw/statcast_history.csv"
    assert store.storage_path("model") == "nrfi/models/nrfi_model.joblib"
    assert SupabaseArtifacts(app_name="hr").storage_path("predictions") == "hr/predictions/latest_predictions.csv"
    assert artifact_spec("training").relative_path == "processed/training_rows.csv"


def test_local_only_mode_when_secrets_absent(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    local_file = tmp_path / "latest_predictions.csv"

    store = SupabaseArtifacts()

    assert store.configured is False
    assert store.download("predictions", local_file) is False
    assert store.upload("predictions", local_file) is False
    assert store.artifact_ready("predictions", local_file) is False


def test_mock_storage_download_and_upload(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")
    local_file = tmp_path / "training_rows.csv"
    fake_client = FakeClient()
    store = SupabaseArtifacts()
    store._client = fake_client

    assert store.download("training", local_file) is True
    assert local_file.read_bytes() == b"remote-bytes"

    local_file.write_text("local,data\n", encoding="utf-8")
    assert store.upload("training", local_file, metadata={"feature_set": "baseline"}) is True
    assert fake_client.bucket.uploaded["nrfi/processed/training_rows.csv"] == b"local,data\n"
    assert fake_client.inserted_manifest["artifact_type"] == "training"
    assert fake_client.inserted_manifest["is_latest"] is True


class FakeClient:
    def __init__(self):
        self.bucket = FakeBucket()
        self.inserted_manifest = {}
        self.storage = FakeStorage(self.bucket)

    def table(self, name):
        return FakeTable(self, name)


class FakeStorage:
    def __init__(self, bucket):
        self.bucket = bucket

    def from_(self, bucket_name):
        return self.bucket


class FakeBucket:
    def __init__(self):
        self.uploaded = {}

    def download(self, path):
        return b"remote-bytes"

    def upload(self, path, data, file_options=None):
        self.uploaded[path] = bytes(data)
        return {}

    def update(self, path, data, file_options=None):
        self.uploaded[path] = bytes(data)
        return {}


class FakeTable:
    def __init__(self, client, name):
        self.client = client
        self.name = name
        self.payload = None

    def select(self, *args, **kwargs):
        return self

    def update(self, *args, **kwargs):
        return self

    def insert(self, payload):
        if self.name == "artifact_manifest":
            self.client.inserted_manifest = payload
        self.payload = payload
        return self

    def upsert(self, payload, *args, **kwargs):
        self.payload = payload
        return self

    def eq(self, *args, **kwargs):
        return self

    def order(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def execute(self):
        return FakeResult([])


class FakeResult:
    def __init__(self, data):
        self.data = data
