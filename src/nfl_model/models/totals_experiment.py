"""Tune totals on 2022–23 and evaluate before combining with saved margin forecasts."""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .ratings import RatingConfig
from .totals import replay_totals,total_score,projected_scores


def run_totals(root: Path) -> Path:
    eff_path=Path(json.loads((root/'models/efficiency/latest.json').read_text())['path'])
    eff=json.loads((eff_path/'manifest.json').read_text())
    base=json.loads((Path(eff['baseline_build'])/'manifest.json').read_text())
    source=base['schedule_source'];path=Path(source['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=source['sha256']:raise ValueError('Schedule checksum mismatch')
    games=pd.read_parquet(path,columns=['season','week','game_id','game_type','gameday','home_team','away_team','home_score','away_score','location'])
    games=games[games.season.between(2021,2025)&games.game_type.eq('REG')]
    trials=[]
    for half in [60,120,240]:
        for ridge in [4,12,32]:
            p,_=replay_totals(games[games.season<=2023],RatingConfig(half,ridge),[2022,2023])
            trials.append(dict(model='team_scoring',half_life_days=half,ridge=ridge,validation_mae=total_score(p,'predicted_total')['mae']))
            if ridge==4:trials.append(dict(model='league_average',half_life_days=half,ridge=ridge,validation_mae=total_score(p,'league_total')['mae']))
    chosen={model:min([t for t in trials if t['model']==model],key=lambda x:x['validation_mae']) for model in ['team_scoring','league_average']}
    winner=min(chosen,key=lambda x:chosen[x]['validation_mae'])
    outputs,states,metrics={},{},{}
    for model,config in chosen.items():
        p,state=replay_totals(games,RatingConfig(config['half_life_days'],config['ridge']),[2022,2023,2024,2025])
        p['selected_total']=p['predicted_total'] if model=='team_scoring' else p['league_total']
        outputs[model]=p;states[model]=state
        metrics[model]={"evaluation":total_score(p[p.season>=2024],'selected_total'),"by_season":{str(y):total_score(g,'selected_total') for y,g in p.groupby('season')}}
    # The success model was selected on tuning years in its own experiment. Keep
    # raw total forecasts separate from feasible combined score projections.
    margins=pd.read_parquet(eff_path/'predictions_success.parquet')[['game_id','predicted_margin','actual_margin']]
    joined=outputs[winner].merge(margins,on='game_id',validate='one_to_one')
    scores=projected_scores(joined.selected_total,joined.predicted_margin)
    joined=pd.concat([joined,scores],axis=1)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]
    folder=root/'models/totals'/stamp;folder.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(trials).to_csv(folder/'tuning.csv',index=False)
    for model,p in outputs.items():p.to_parquet(folder/f'predictions_{model}.parquet',index=False)
    joined.to_parquet(folder/'projected_scores.parquet',index=False)
    (folder/'weekly_models.json').write_text(json.dumps(states,indent=2),encoding='utf-8')
    run={"version":stamp,"selected_on_validation":winner,"configs":chosen,"metrics":metrics,"source":source,"efficiency_build":str(eff_path),
         "score_constraints_applied":int(joined.total_constraint_applied.sum()),"code_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    (folder/'manifest.json').write_text(json.dumps(run,indent=2),encoding='utf-8')
    lines=['# Totals baseline and projected scores','',f'Selected using 2022–2023 only: **{winner}**.','',
           '2021 initializes history. Team scoring effects are fit to points scored against opposing defenses, with time decay and ridge shrinkage. A simpler decayed league-average total is tuned separately. All fits use earlier completed weeks only. No sportsbook fields enter the model.','',
           '| Model | Tuning MAE | 2024 MAE | 2025 MAE | Evaluation MAE | Evaluation RMSE |','|---|---:|---:|---:|---:|---:|']
    for model,m in metrics.items():lines.append(f"| {model} | {chosen[model]['validation_mae']:.3f} | {m['by_season']['2024']['mae']:.3f} | {m['by_season']['2025']['mae']:.3f} | {m['evaluation']['mae']:.3f} | {m['evaluation']['rmse']:.3f} |")
    lines+=['','## Projected scores','',f"Combined selected totals with the saved success-model home margin: home=(total+margin)/2; away=(total-margin)/2. {int(joined.total_constraint_applied.sum())} projections needed total raised to absolute margin to avoid negative scores. Raw totals remain available separately.",'',
            'These are continuous expected-score estimates, not promises of exact integer scores. Both evaluation models cover the same 544 games in 2024–2025. These years are a reused evaluation set, not fresh validation. Historical publication timing remains uncertified. No win-probability calibration has been added in this stage. Existing frozen Week 2 margin forecasts are unchanged.','',f'Artifacts: {folder.resolve()}']
    report='\n'.join(lines);(folder/'report.md').write_text(report,encoding='utf-8');Path('docs/totals_experiment.md').write_text(report,encoding='utf-8')
    temp=folder.parent/f'latest_{uuid.uuid4().hex}.tmp';temp.write_text(json.dumps({'path':str(folder.resolve()),'version':stamp}),encoding='utf-8');temp.replace(folder.parent/'latest.json')
    return folder
