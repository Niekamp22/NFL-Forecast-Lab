"""Chronological margin calibration with explicit home/tie/away outcomes."""
from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOMES = ["away", "tie", "home"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def fit_probability(history: pd.DataFrame, penalty: float) -> dict:
    """Fit decisive-game logistic calibration with L2 shrinkage.

    Predictors are intercept and pregame home margin / 10. A nonnegative slope
    ensures a stronger home margin cannot lower its win probability. Tie rate
    uses an explicit 100-game, 1%-tie smoothing prior; it is not margin-specific.
    """
    if penalty <= 0 or history.empty:
        raise ValueError("Need history and positive calibration penalty")
    if history.actual_margin.isna().any() or not np.isfinite(history.predicted_margin).all():
        raise ValueError("Invalid calibration history")
    decisive = history[history.actual_margin.ne(0)]
    if decisive.empty:
        raise ValueError("No decisive training games")
    x = np.column_stack([np.ones(len(decisive)), decisive.predicted_margin.to_numpy()/10])
    y = decisive.actual_margin.gt(0).to_numpy(dtype=float)
    beta = np.array([0., 1.])
    prior = np.array([0., 1.])
    for _ in range(100):
        p = sigmoid(x @ beta)
        gradient = x.T @ (p-y) + penalty*(beta-prior)
        hessian = x.T @ ((p*(1-p))[:,None]*x) + np.eye(2)*penalty
        step = np.linalg.solve(hessian, gradient)
        next_beta = beta-step
        next_beta[1] = max(0., next_beta[1])
        if np.max(np.abs(next_beta-beta)) < 1e-9:
            beta = next_beta
            break
        beta = next_beta
    counts = np.array([(history.actual_margin<0).sum(), (history.actual_margin==0).sum(), (history.actual_margin>0).sum()], dtype=float)
    rates = (counts+np.array([49.5,1.,49.5]))/(len(history)+100)
    return {"coefficients": beta.tolist(), "tie_probability": float(rates[1]), "baseline_probabilities": rates.tolist(),
            "training_games": len(history), "decisive_games": len(decisive), "penalty": penalty,
            "tie_prior_games":100, "tie_prior_rate":.01}


def predict_probability(margins: np.ndarray, model: dict) -> np.ndarray:
    """Return [away win, tie, home win], summing to one for every game."""
    margins = np.asarray(margins, dtype=float)
    if not np.isfinite(margins).all():
        raise ValueError("Non-finite prediction margin")
    conditional = sigmoid(model["coefficients"][0]+model["coefficients"][1]*margins/10)
    tie = np.full(len(margins), model["tie_probability"])
    return np.column_stack([(1-tie)*(1-conditional),tie,(1-tie)*conditional])


def replay_probabilities(frame: pd.DataFrame, penalty: float, seasons: list[int]) -> tuple[pd.DataFrame,list[dict]]:
    """Fit calibration before each week using only earlier completed predictions."""
    if frame.game_id.duplicated().any():
        raise ValueError("Duplicate games in calibration input")
    pieces, states = [], []
    for (season,week), target in frame[frame.season.isin(seasons)].groupby(["season","week"],sort=True):
        cutoff = pd.to_datetime(target.cutoff_date).min()
        earlier = (frame.season<season)|((frame.season==season)&(frame.week<week))
        history = frame[earlier & frame.actual_margin.notna() & (pd.to_datetime(frame.gameday)<cutoff)]
        model = fit_probability(history,penalty)
        probabilities = predict_probability(target.predicted_margin.to_numpy(),model)
        result = target.copy()
        for i,name in enumerate(OUTCOMES):
            result[f"p_{name}"] = probabilities[:,i]
            result[f"baseline_p_{name}"] = model["baseline_probabilities"][i]
        pieces.append(result)
        states.append({"season":int(season),"week":int(week),"cutoff_date":str(cutoff),"last_training_date":str(history.gameday.max()),**model})
    return pd.concat(pieces,ignore_index=True),states


def probability_score(frame:pd.DataFrame,prefix:str="p_")->dict:
    complete=frame[frame.actual_margin.notna()]
    if complete.empty:
        return {"games":0,"log_loss":None,"brier_multiclass":None,"home_brier":None}
    probabilities=complete[[prefix+n for n in OUTCOMES]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities<0).any() or (probabilities>1).any() or not np.allclose(probabilities.sum(axis=1),1):
        raise ValueError("Invalid three-way probabilities")
    classes=np.sign(complete.actual_margin.to_numpy()).astype(int)+1
    truth=np.eye(3)[classes]
    return {"games":len(complete),"log_loss":float(-np.log(np.clip(probabilities[np.arange(len(complete)),classes],1e-15,1)).mean()),
            "brier_multiclass":float(((probabilities-truth)**2).sum(axis=1).mean()),
            "home_brier":float(((probabilities[:,2]-truth[:,2])**2).mean())}


def reliability(frame:pd.DataFrame)->pd.DataFrame:
    """Fixed decile bins, counts and observed win rates; ties count as non-wins."""
    rows=[]
    complete=frame[frame.actual_margin.notna()]
    for outcome in OUTCOMES:
        p=complete[f"p_{outcome}"]
        win={"home":complete.actual_margin.gt(0),"away":complete.actual_margin.lt(0),"tie":complete.actual_margin.eq(0)}[outcome]
        for i in range(10):
            selected=p.ge(i/10)&(p.le(1) if i==9 else p.lt((i+1)/10))
            rows.append({"outcome":outcome,"bin_lower":i/10,"bin_upper":(i+1)/10,"games":int(selected.sum()),
                         "mean_probability":float(p[selected].mean()) if selected.any() else None,
                         "observed_rate":float(win[selected].mean()) if selected.any() else None})
    return pd.DataFrame(rows)
