"""Load probabilities only when they match the exact selected frozen forecast."""
import hashlib
import json
from pathlib import Path
import pandas as pd
from .artifact_paths import resolve_artifact


def load_milestones(root,season,week,defense,forecast_folder,forecast_meta):
    family='defense' if defense else 'baseline'
    candidates=[]
    for path in (Path(root)/'player_milestones'/family/str(season)/f'week_{week:02d}').glob('*/manifest.json'):
        meta=json.loads(path.read_text())
        if (meta['forecast_sha256']==forecast_meta['predictions_sha256']
            and resolve_artifact(meta['forecast_path']).resolve()==Path(forecast_folder).resolve()
            and pd.Timestamp(meta['created_at'])<pd.Timestamp(meta['earliest_kickoff'])):
            candidates.append((pd.Timestamp(meta['created_at']),str(path),meta))
    if not candidates:return None
    _,path,meta=min(candidates,key=lambda x:x[:2])
    file=Path(path).parent/'probabilities.parquet'
    if hashlib.sha256(file.read_bytes()).hexdigest()!=meta['probabilities_sha256']:raise ValueError('Milestone probability checksum mismatch')
    model=resolve_artifact(meta['model_path']);model_file=model/'manifest.json'
    if hashlib.sha256(model_file.read_bytes()).hexdigest()!=meta['model_manifest_sha256']:raise ValueError('Milestone model checksum mismatch')
    model_meta=json.loads(model_file.read_text())
    for name in ['metrics.csv','reliability.csv']:
        if hashlib.sha256((model/name).read_bytes()).hexdigest()!=model_meta['artifacts'][name]:raise ValueError('Milestone evaluation checksum mismatch')
    return pd.read_parquet(file),pd.read_csv(model/'metrics.csv'),pd.read_csv(model/'reliability.csv'),meta
