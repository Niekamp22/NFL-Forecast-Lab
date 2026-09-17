"""Manual weekly workflow with immutable forecasts and repeatable result updates."""
import json
from pathlib import Path

from .data.client import NFLVerseClient
from .data.build import build_database
from .data.qb_evidence import build_qb_evidence
from .models.prospective import freeze_week
from .models.score_forecast import freeze_scores
from .models.weekly_forecast import freeze_weekly_forecast,earliest_snapshot,checked_predictions
from .models.full_grading import grade_full_week


def run_player_week(client,season,week,refresh=False):
    """Prepare missing player forecasts once; grade immutable forecasts on reruns."""
    from .models.player_roles import freeze_role_forecast
    from .models.player_projection import build_player_projections
    from .models.player_grading import grade_role_week
    root=client.root
    role=root/'player_role_forecasts'/str(season)/f'week_{week:02d}'
    opportunity=root/'player_opportunity_forecasts'/str(season)/f'week_{week:02d}'
    if not list(role.glob('*/manifest.json')) and not list(opportunity.glob('*/manifest.json')):
        build_player_projections(client,season,week,opportunity_model=True)
    forecast=freeze_role_forecast(root,season,week)
    grading=grade_role_week(client,season,week,refresh)
    return {'forecast':str(forecast),'grading':str(grading)}


def run_week(client:NFLVerseClient,season:int,week:int,refresh:bool=False)->dict:
    """Prepare missing artifacts once, then grade the first unified forecast.

    Model research/tuning is never rerun here. Existing forecasts are reused even
    if newer sources become available; new results create new grading records.
    """
    if week<1 or week>18:
        raise ValueError('REG week must be between 1 and 18')
    root=client.root
    for model in ['baseline','efficiency','totals','probability']:
        if not (root/'models'/model/'latest.json').exists():
            raise ValueError(f'Missing {model} model; run the documented research command first')
    target=root/'forecasts'/str(season)/f'week_{week:02d}'
    if list(target.glob('*/manifest.json')):
        forecast,meta=earliest_snapshot(target)
        checked_predictions(forecast,meta)
        grading=grade_full_week(client,season,week,refresh)
        return {'forecast':str(forecast),'forecast_action':'reused earliest frozen run','grading':str(grading)}
    margin_root=root/'snapshots'/str(season)/f'week_{week:02d}'
    if not list(margin_root.glob('*/manifest.json')):
        build_database(client,[season],force=refresh)
        build_qb_evidence(root,season,week)
        freeze_week(root,season,week)
    score_root=root/'snapshots_scores'/str(season)/f'week_{week:02d}'
    if not list(score_root.glob('*/manifest.json')):
        freeze_scores(root,season,week)
    forecast=freeze_weekly_forecast(root,season,week)
    grading=grade_full_week(client,season,week,refresh)
    return {'forecast':str(forecast),'forecast_action':'created before kickoff','grading':str(grading)}
