"""Download untouched assets with immutable retrieval versions and integrity checks."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .catalog import DATASETS

LOG = logging.getLogger(__name__)
API = "https://api.github.com/repos/nflverse/nflverse-data/releases"


class DataUnavailableError(RuntimeError):
    """The requested asset or season is not present; never substitute a year."""


class NFLVerseClient:
    def __init__(self, root: str | Path = "data", session=None):
        self.root = Path(root)
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "nfl-model-data-foundation/0.1"})
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
        self._releases: dict = {}

    def release(self, tag: str, refresh: bool = False) -> dict:
        """Discover actual filenames, preserving the response as retrieval evidence."""
        if tag not in self._releases or refresh:
            response = self.session.get(f"{API}/tags/{tag}", timeout=(15, 90))
            response.raise_for_status()
            self._releases[tag] = response.json()
            folder = self.root / "raw" / "_catalog"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{tag}_{uuid.uuid4().hex}.json").write_text(json.dumps({
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "release": self._releases[tag]}, indent=2), encoding="utf-8")
        return self._releases[tag]

    def available_seasons(self, dataset: str) -> list[int]:
        spec = DATASETS[dataset]
        if "{season}" not in spec.filename:
            return []  # Combined files require row inspection.
        pattern = re.escape(spec.filename).replace(re.escape("{season}"), r"(\d{4})")
        return sorted({int(m[1]) for a in self.release(spec.tag)["assets"]
                       if (m := re.fullmatch(pattern, a["name"]))})

    def download(self, dataset: str, season: int | None = None, force: bool = False) -> Path:
        """Use a verified cache by default; force creates a new immutable version."""
        spec = DATASETS[dataset]
        if "{season}" in spec.filename and season is None:
            raise ValueError(f"{dataset} requires a season")
        name = spec.filename.format(season=season)
        folder = self.root / "raw" / dataset / (str(season) if "{season}" in spec.filename else "all")
        pointer = folder / "latest.json"
        if pointer.exists() and not force:
            meta = json.loads(pointer.read_text(encoding="utf-8"))
            cached = folder / meta["version"] / name
            if cached.exists() and hashlib.sha256(cached.read_bytes()).hexdigest() == meta["sha256"]:
                LOG.info("Cache hit %s", cached)
                return cached
            LOG.warning("Cache missing or corrupt: %s; downloading again", cached)
        assets = self.release(spec.tag, refresh=force)["assets"]
        asset = next((a for a in assets if a["name"] == name), None)
        if asset is None:
            raise DataUnavailableError(f"{dataset}: {name} absent from live release {spec.tag}")
        version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
        target = folder / version / name
        target.parent.mkdir(parents=True, exist_ok=False)
        partial = target.with_suffix(".partial")
        try:
            with self.session.get(asset["browser_download_url"], stream=True, timeout=(15, 180)) as response:
                response.raise_for_status()
                with partial.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
                if partial.stat().st_size == 0 or partial.stat().st_size != asset["size"]:
                    raise ValueError(f"Invalid download size for {name}")
                pq.read_metadata(partial)
                digest = hashlib.sha256(partial.read_bytes()).hexdigest()
                if asset.get("digest", "") and asset["digest"].startswith("sha256:") and asset["digest"] != f"sha256:{digest}":
                    raise ValueError(f"Source checksum mismatch for {name}")
                meta = {"dataset": dataset, "season": season, "version": version,
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "source_updated_at": asset.get("updated_at"), "asset_id": asset["id"],
                        "url": asset["browser_download_url"], "sha256": digest,
                        "bytes": partial.stat().st_size, "etag": response.headers.get("ETag")}
            partial.replace(target)
            (target.parent / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            temp_pointer = folder / f"latest_{uuid.uuid4().hex}.tmp"
            temp_pointer.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            temp_pointer.replace(pointer)
            LOG.info("Downloaded %s (%s bytes)", name, meta["bytes"])
            return target
        except Exception:
            partial.unlink(missing_ok=True)
            LOG.exception("Download failed for %s", name)
            raise

    def load(self, dataset: str, seasons: list[int] | None = None, force: bool = False) -> pd.DataFrame:
        """Load requested seasons strictly. An omitted season list discovers all annual assets."""
        spec = DATASETS[dataset]
        if seasons == []:
            raise ValueError("seasons cannot be empty")
        if "{season}" in spec.filename:
            selected = seasons if seasons is not None else self.available_seasons(dataset)
            if not selected:
                raise DataUnavailableError(dataset)
            frames = [pd.read_parquet(self.download(dataset, s, force)) for s in sorted(set(selected))]
            frame = pd.concat(frames, ignore_index=True)
        else:
            frame = pd.read_parquet(self.download(dataset, force=force))
            if seasons is not None:
                field = next((c for c in ("season", "year", "draft_year") if c in frame), None)
                if field is None:
                    raise ValueError(f"{dataset} has no season field; load without seasons")
                frame = frame[frame[field].isin(seasons)].copy()
        if frame.empty:
            raise DataUnavailableError(f"{dataset}: no rows for requested seasons {seasons}")
        return frame
