"""Evaluate calibration honestly before publishing three-way probabilities."""
import hashlib
import json
import uuid
from datetime import datetime,timezone
from pathlib import Path

import pandas as pd

from .probability import replay_probabilities,probability_score,reliability


def run_probability_experiment(root:Path)->Path:
    source=Path(json.loads((root/'models/efficiency/latest.json').read_text())['path'])
    frame=pd.read_parquet(source/'predictions_success.parquet')
    if set(frame.season.unique())!={2022,2023,2024,2025}:raise ValueError('Expected 2022–2025 pregame forecasts')
    trials=[]
    for penalty in [1.,10.,100.]:
        validation,_=replay_probabilities(frame[frame.season<=2023],penalty,[2023])
        trials.append({'penalty':penalty,**probability_score(validation)})
    selected=min(trials,key=lambda row:row['log_loss'])
    predictions,states=replay_probabilities(frame,selected['penalty'],[2023,2024,2025])
    evaluation=predictions[predictions.season>=2024]
    metrics={'calibrated':probability_score(evaluation),'frequency_baseline':probability_score(evaluation,'baseline_p_'),
             'by_season':{str(y):probability_score(g) for y,g in predictions.groupby('season')}}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]
    folder=root/'models/probability'/stamp;folder.mkdir(parents=True,exist_ok=False)
    predictions.to_parquet(folder/'predictions.parquet',index=False)
    bins=reliability(evaluation);bins.to_csv(folder/'reliability.csv',index=False)
    (folder/'weekly_models.json').write_text(json.dumps(states,indent=2),encoding='utf-8')
    pd.DataFrame(trials).to_csv(folder/'tuning.csv',index=False)
    info={'version':stamp,'efficiency_build':str(source),'selected':selected,'metrics':metrics,
          'calibration_warmup':[2022],'calibration_tuning':[2023],'reused_evaluation':[2024,2025],
          'input_sha256':hashlib.sha256((source/'predictions_success.parquet').read_bytes()).hexdigest(),
          'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    (folder/'manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    lines=['# Win-probability calibration','',
           'The margin model was tuned on 2022–2023. Calibration starts with its 2022 chronological predictions; penalty is selected on 2023 log loss only. Calibration refits weekly using strictly earlier completed games. 2024–2025 are reused evaluation years, not fresh evidence.','',
           'The logistic curve maps home margin to home-win probability conditional on a decisive result. A separately smoothed historical tie rate produces home/tie/away probabilities summing to one. The tie prior contributes 100 games at a 1% tie rate; this is an explicit sparse-data approximation, not a calibrated game-specific tie model.','',
           '## Evaluation (lower is better)','', '| Model | Games | Log loss | Three-way Brier | Home-win Brier |','|---|---:|---:|---:|---:|']
    for label in ['calibrated','frequency_baseline']:
        m=metrics[label];lines.append(f"| {label} | {m['games']} | {m['log_loss']:.4f} | {m['brier_multiclass']:.4f} | {m['home_brier']:.4f} |")
    lines+=['','The baseline uses only earlier home/tie/away frequencies with the same smoothing prior. Log loss penalizes confidently wrong forecasts; Brier measures squared probability error. These comparisons do not prove the margin probabilities are perfectly calibrated.','',
            '## Home-win reliability','', 'Each bin compares average forecast probability with observed home-win rate. Small bins are noisy.','', '```text',bins[bins.outcome.eq('home') & bins.games.gt(0)].to_string(index=False),'```','',
            'No sportsbook inputs or betting recommendations. No future labels enter calibration. QB uncertainty is displayed separately and is not modeled in these probabilities.']
    report='\n'.join(lines);(folder/'report.md').write_text(report,encoding='utf-8');Path('docs/probability_calibration.md').write_text(report,encoding='utf-8')
    tmp=folder.parent/f'latest_{uuid.uuid4().hex}.tmp';tmp.write_text(json.dumps({'path':str(folder.resolve()),'version':stamp}),encoding='utf-8');tmp.replace(folder.parent/'latest.json')
    return folder
