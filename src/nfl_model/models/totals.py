"""Opponent-adjusted scoring baseline with chronological weekly fits."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ratings import RatingConfig


def fit_scoring(history: pd.DataFrame, teams: list[str], cutoff: pd.Timestamp, config: RatingConfig) -> tuple[np.ndarray, float]:
    """Fit points scored = intercept + own offense + opposing points-allowed effect.

    A home-advantage term (+0.5 home, -0.5 away; zero at neutral sites) models
    scoring asymmetry but cancels in the combined total. Team effects are shrunk.
    """
    if history.empty or config.ridge <= 0 or config.half_life_days <= 0:
        raise ValueError("Need prior games and positive regularization/half life")
    dates = pd.to_datetime(history.gameday)
    if (dates >= cutoff).any():
        raise ValueError("Training games at or after cutoff")
    if history[["home_score", "away_score"]].isna().any().any():
        raise ValueError("Training scores incomplete")
    n = len(teams)
    ids = {team: i for i, team in enumerate(teams)}
    x = np.zeros((2*len(history), 2*n+2))
    y = np.zeros(2*len(history))
    for i, game in enumerate(history.itertuples()):
        for offset, own, opponent, points, venue in [(0,game.home_team,game.away_team,game.home_score,.5), (1,game.away_team,game.home_team,game.away_score,-.5)]:
            row = 2*i+offset
            x[row, 0] = 1
            x[row, 1+ids[own]] = 1
            x[row, 1+n+ids[opponent]] = 1
            x[row, -1] = venue if game.location == "Home" else 0
            y[row] = points
    weights = np.exp2(-(cutoff-dates).dt.total_seconds().to_numpy()/86400/config.half_life_days)
    doubled = np.repeat(weights,2)
    penalty = np.eye(x.shape[1])*config.ridge
    penalty[0,0] = penalty[-1,-1] = 1e-8
    coefficients = np.linalg.solve(x.T@(doubled[:,None]*x)+penalty, x.T@(doubled*y))
    league_total = float(np.average(history.home_score+history.away_score,weights=weights))
    return coefficients, league_total


def replay_totals(games: pd.DataFrame, config: RatingConfig, seasons: list[int]) -> tuple[pd.DataFrame,list[dict]]:
    """Freeze each week's predictions using only earlier completed weeks."""
    if games.game_id.duplicated().any() or not games.location.isin(["Home","Neutral"]).all():
        raise ValueError("Duplicate games or unsupported venue")
    teams = sorted(set(games.home_team)|set(games.away_team)); ids={t:i for i,t in enumerate(teams)}
    predictions,states=[],[]
    for (season,week),target in games[games.season.isin(seasons)].groupby(["season","week"],sort=True):
        cutoff=pd.to_datetime(target.gameday).min()
        earlier=(games.season<season)|((games.season==season)&(games.week<week))
        history=games[earlier & games.home_score.notna() & games.away_score.notna() & (pd.to_datetime(games.gameday)<cutoff)]
        beta,league=fit_scoring(history,teams,cutoff,config)
        states.append({"season":int(season),"week":int(week),"cutoff":str(cutoff),"last_training_date":str(history.gameday.max()),
                       "training_games":len(history),"teams":teams,"coefficients":beta.tolist(),"league_total":league})
        for g in target.itertuples():
            i,j=ids[g.home_team],ids[g.away_team]
            total=2*beta[0]+beta[1+i]+beta[1+j]+beta[1+len(teams)+i]+beta[1+len(teams)+j]
            predictions.append(dict(season=season,week=week,game_id=g.game_id,home_team=g.home_team,away_team=g.away_team,
                                    predicted_total=max(0.,float(total)),league_total=league,actual_total=g.home_score+g.away_score))
    return pd.DataFrame(predictions),states


def total_score(frame: pd.DataFrame, column: str) -> dict:
    complete=frame[frame.actual_total.notna()]
    error=complete[column]-complete.actual_total
    return {"games":len(complete),"mae":float(error.abs().mean()),"rmse":float(np.sqrt((error**2).mean())),"mean_error":float(error.mean())}


def projected_scores(total: pd.Series, margin: pd.Series) -> pd.DataFrame:
    """Preserve margin; raise infeasible total to |margin| and expose the constraint."""
    if not np.isfinite(total).all() or not np.isfinite(margin).all() or (total<0).any():
        raise ValueError("Invalid total or margin")
    feasible=np.maximum(total,margin.abs())
    return pd.DataFrame({"score_total":feasible,"total_constraint_applied":total<margin.abs(),
                         "projected_home_score":(feasible+margin)/2,"projected_away_score":(feasible-margin)/2},index=total.index)
