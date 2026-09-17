"""One pinned weekly forecast combining margins, scores and probabilities."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .probability import fit_probability,predict_probability,OUTCOMES


def earliest_snapshot(root:Path)->tuple[Path,dict]:
    """Choose earliest valid pre-kickoff snapshot, never a favorable later revision."""
    candidates=[]
    for path in root.glob('*/manifest.json'):
        m=json.loads(path.read_text())
        if pd.Timestamp(m['created_at'])<pd.Timestamp(m['earliest_kickoff']):
            candidates.append((pd.Timestamp(m['created_at']),str(path.parent),m))
    if not candidates:raise ValueError(f'No valid frozen snapshot in {root}')
    _,path,meta=min(candidates,key=lambda item:(item[0],item[1]))
    return Path(path),meta


def checked_predictions(folder:Path,meta:dict)->pd.DataFrame:
    path=folder/'predictions.parquet'
    expected=meta.get('predictions_sha256',meta.get('prediction_sha256'))
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if expected and expected!=actual:raise ValueError(f'Prediction integrity failure: {folder}')
    frame=pd.read_parquet(path)
    if frame.game_id.duplicated().any() or len(frame)!=meta['games']:
        raise ValueError('Invalid forecast game count or duplicate IDs')
    return frame


def freeze_weekly_forecast(root:Path,season:int,week:int)->Path:
    """Create an additive frozen artifact; no older predictions are modified."""
    now=pd.Timestamp(datetime.now(timezone.utc))
    margin_path,margin_meta=earliest_snapshot(root/'snapshots'/str(season)/f'week_{week:02d}')
    score_path,score_meta=earliest_snapshot(root/'snapshots_scores'/str(season)/f'week_{week:02d}')
    if Path(score_meta['margin_snapshot']).resolve()!=margin_path.resolve():raise ValueError('Score and margin ancestry differ')
    kickoff=min(pd.Timestamp(margin_meta['earliest_kickoff']),pd.Timestamp(score_meta['earliest_kickoff']))
    if now>=kickoff:raise ValueError('Cannot freeze weekly forecasts after kickoff')
    margins=checked_predictions(margin_path,margin_meta)
    scores=checked_predictions(score_path,score_meta)
    if set(margins.game_id)!=set(scores.game_id):raise ValueError('Margin/score cohorts differ')
    frame=margins.merge(scores,on=['game_id','home_team','away_team','success_model_margin'],validate='one_to_one')
    if len(frame)!=len(margins):raise ValueError('Margin/score values or teams differ')
    calibration_path=Path(json.loads((root/'models/probability/latest.json').read_text())['path'])
    calibration=json.loads((calibration_path/'manifest.json').read_text())
    if Path(calibration['efficiency_build']).resolve()!=Path(margin_meta['efficiency_build']).resolve():
        raise ValueError('Calibration and margin models differ')
    history_path=Path(calibration['efficiency_build'])/'predictions_success.parquet'
    if hashlib.sha256(history_path.read_bytes()).hexdigest()!=calibration['input_sha256']:raise ValueError('Calibration input checksum mismatch')
    history=pd.read_parquet(history_path)
    history=history[(history.season<season)&history.actual_margin.notna()&(pd.to_datetime(history.gameday,utc=True)<now)]
    model=fit_probability(history,calibration['selected']['penalty'])
    probs=predict_probability(frame.success_model_margin.to_numpy(),model)
    for i,outcome in enumerate(OUTCOMES):frame[f'p_{outcome}']=probs[:,i]
    frame['qb_adjustment_applied']=False
    # Evidence is contextual, from the latest separately recorded report, not a predictor.
    evidence_paths=[]
    for p in (root/'processed/qb_evidence').glob('*/manifest.json'):
        m=json.loads(p.read_text())
        if m['season']==season and m['week']==week and pd.Timestamp(m['created_at'])<=now:
            evidence_paths.append((pd.Timestamp(m['created_at']),p.parent))
    evidence_path=max(evidence_paths,key=lambda item:item[0])[1] if evidence_paths else None
    flags={}
    if evidence_path:
        evidence=pd.read_parquet(evidence_path/'qb_evidence.parquet')
        for team,g in evidence.groupby('team'):
            top=g[g.listed_qb_rank.eq(1)]
            tags=[]
            if top.empty:tags.append('no listed QB1')
            if g.review_flags.str.contains('depth_usage_disagree').any():tags.append('QB depth/usage disagreement')
            if top.review_flags.str.contains('injury_report_unavailable').any():tags.append('QB injury report unavailable')
            if top.review_flags.str.contains('injury_designation|practice_restriction|non_active_roster_status').any():tags.append('QB availability concern')
            flags[team]='; '.join(tags) or 'Starter not confirmed'
    frame['home_qb_review']=frame.home_team.map(flags).fillna('QB evidence unavailable')
    frame['away_qb_review']=frame.away_team.map(flags).fillna('QB evidence unavailable')
    source=score_meta['source'];schedule_path=Path(source['path'])
    if hashlib.sha256(schedule_path.read_bytes()).hexdigest()!=source['sha256']:raise ValueError('Schedule checksum mismatch')
    schedule=pd.read_parquet(schedule_path,columns=['game_id','gameday','gametime','home_score','away_score'])
    schedule=schedule[schedule.game_id.isin(frame.game_id)]
    if schedule.game_id.duplicated().any() or set(schedule.game_id)!=set(frame.game_id):raise ValueError('Schedule coverage mismatch')
    if schedule[['home_score','away_score']].notna().any().any():raise ValueError('Target results already present')
    frame=frame.merge(schedule[['game_id','gameday','gametime']],on='game_id',validate='one_to_one')
    frame['kickoff_utc']=pd.to_datetime(frame.gameday.astype(str)+' '+frame.gametime.astype(str)).dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    if frame.kickoff_utc.isna().any() or (frame.kickoff_utc<=now).any():raise ValueError('Invalid or past kickoff')
    if not np.allclose(frame[['p_away','p_tie','p_home']].sum(axis=1),1):raise ValueError('Probability normalization failure')
    stamp=now.strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]
    folder=root/'forecasts'/str(season)/f'week_{week:02d}'/stamp;folder.mkdir(parents=True,exist_ok=False)
    frame.to_parquet(folder/'predictions.parquet',index=False);frame.to_csv(folder/'predictions.csv',index=False)
    refs={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [margin_path/'predictions.parquet',score_path/'predictions.parquet',calibration_path/'manifest.json',history_path]}
    if evidence_path:
        for filename in ['qb_evidence.parquet','reviewed_news.json','manifest.json']:
            path=evidence_path/filename
            if path.exists():refs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest={'created_at':str(now),'earliest_kickoff':str(frame.kickoff_utc.min()),'season':season,'week':week,'games':len(frame),
              'margin_snapshot':str(margin_path),'score_snapshot':str(score_path),'calibration_build':str(calibration_path),
              'qb_evidence':str(evidence_path) if evidence_path else None,'source':source,'input_hashes':refs,
              'calibrator':model,'calibration_last_game_date':str(history.gameday.max()),'calibration_policy':'fit to prior seasons only; 2026 prospective outcomes will be evaluated before any recalibration policy change',
              'legacy_margin_creation_hash_available':bool(margin_meta.get('predictions_sha256')),
              'predictions_sha256':hashlib.sha256((folder/'predictions.parquet').read_bytes()).hexdigest(),
              'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    lines=[f'# NFL weekly forecast — {season} Week {week}','',f'Frozen {now}. {len(frame)} games. Positive margin means the home team is favored.','',
           '| Away | Home | Away score | Home score | Total | Home margin | Away win | Tie | Home win |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in frame.sort_values(['kickoff_utc','game_id']).itertuples():
        lines.append(f'| {r.away_team} | {r.home_team} | {r.projected_away_score:.1f} | {r.projected_home_score:.1f} | {r.score_total:.1f} | {r.success_model_margin:+.1f} | {r.p_away:.1%} | {r.p_tie:.1%} | {r.p_home:.1%} |')
    lines+=['','## QB and availability review','', 'Probabilities and scores do not adjust for unresolved QB situations. These rows need review before treating team strength as representative of the current lineup.','']
    for r in frame.itertuples():lines.append(f'- {r.away_team} at {r.home_team}: {r.away_team}: {r.away_qb_review}; {r.home_team}: {r.home_qb_review}.')
    lines+=['','## Provenance and limits','',f'- Schedule updated {source.get("source_updated_at")}; retrieved {source["retrieved_at"]}.',
            f'- Margin snapshot: {margin_path.name}; score snapshot: {score_path.name}.',
            f'- Calibration uses {len(history)} completed historical forecasts, through {history.gameday.max()}.',
            '- Home/tie/away probabilities sum to one before display rounding. Tie probability is a smoothed league rate, not a game-specific estimate.',
            '- The historical evaluation has been reused during research. Probability calibration is estimated, not guaranteed.',
            '- Original forecasts remain unchanged. No sportsbook odds, value comparisons or stake recommendations are included.',
            '- The first margin snapshot predates creation-hash support; its hash was pinned later. This limitation remains in the manifest.']
    report='\n'.join(lines);(folder/'report.md').write_text(report,encoding='utf-8');Path(f'docs/weekly_forecast_{season}_week_{week:02d}.md').write_text(report,encoding='utf-8')
    return folder
