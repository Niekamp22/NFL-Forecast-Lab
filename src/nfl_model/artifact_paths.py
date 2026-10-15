"""Resolve original immutable artifact references in a portable review bundle."""
import json
import os
from pathlib import Path


def _mapping(filename):
    # Deployments replace this file in place while Streamlit can retain imports.
    # A filename-only cache keeps the previous release's allowlist indefinitely.
    return json.loads(Path(filename).read_text(encoding='utf-8'))


def resolve_artifact(value):
    mapping_file = os.environ.get('NFL_ARTIFACT_MAP')
    if not mapping_file:
        return Path(value)
    mapping = _mapping(mapping_file)
    key = str(value).replace('\\', '/')
    if key not in mapping:
        raise ValueError('Artifact is not included in the review bundle')
    root = Path(mapping_file).resolve().parent
    result = (root / mapping[key]).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Artifact path escapes the review bundle')
    return result
