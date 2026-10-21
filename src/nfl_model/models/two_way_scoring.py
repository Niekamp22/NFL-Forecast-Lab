"""Independent player signals -> team scores -> jointly reconciled scoring events.

Research challenger. Inputs must be raw allocations before the shared-score
transform; feeding reconciled TDs back as independent evidence is prohibited.
"""
import numpy as np
import pandas as pd
from .joint_scoring import reconcile_scoring

PLAYER_FEATURES=['player_points_gap','passing_yards','rushing_yards','attempts','carries',
                 'interceptions','qb_resolved','unassigned_target_share','unassigned_carry_share','low_history_share']
COMPONENT_FEATURES=['pass_td_points','rush_td_points','fg_points','pat_points']


def player_score_features(players,games,other_points):
    if 'scoring_team_points' in players and players.scoring_team_points.notna().any():
        raise ValueError('Two-way scoring requires independent unreconciled player estimates')
    if not np.isfinite(other_points) or other_points<0:raise ValueError('Invalid other-points allowance')
    if games.game_id.duplicated().any():raise ValueError('Duplicate team forecast games')
    rows=[]
    for (gid,team),g in players.groupby(['game_id','team']):
        match=games[games.game_id.eq(gid)]
        if len(match)!=1:raise ValueError('Missing team score forecast')
        game=match.iloc[0]
        if team not in [game.home_team,game.away_team]:raise ValueError('Player team does not match game')
        reserve=g[g.is_unallocated];named=g[~g.is_unallocated]
        if len(reserve)!=1:raise ValueError('Require one explicit team reserve')
        home=team==game.home_team
        base=float(game.projected_home_score if home else game.projected_away_score)
        pass_td=6*float(g.receiving_tds.sum());rush_td=6*float(g.rushing_tds.sum())
        fg=3*float(g.fg_made.sum());pat=float(g.pat_made.sum())
        volume=float(g.targets.sum()+g.carries.sum())
        history=named.workload_history_games if 'workload_history_games' in named else named.history_games
        low=named[history.lt(3)]
        rows.append(dict(game_id=gid,team=team,opponent=game.away_team if home else game.home_team,is_home=float(home),baseline_score=base,
            player_points=pass_td+rush_td+fg+pat+other_points,player_points_gap=pass_td+rush_td+fg+pat+other_points-base,
            pass_td_points=pass_td,rush_td_points=rush_td,fg_points=fg,pat_points=pat,other_points=float(other_points),
            passing_yards=float(g.receiving_yards.sum()),rushing_yards=float(g.rushing_yards.sum()),attempts=float(g.attempts.sum()),carries=float(g.carries.sum()),
            interceptions=float(g.passing_interceptions.sum()),
            qb_resolved=float(named.passing_yards.notna().any()),
            unassigned_target_share=float(reserve.targets.sum()/g.targets.sum()) if g.targets.sum()>0 else 0.,
            unassigned_carry_share=float(reserve.carries.sum()/g.carries.sum()) if g.carries.sum()>0 else 0.,
            low_history_share=float((low.targets.sum()+low.carries.sum())/volume) if volume>0 else 1.))
    frame=pd.DataFrame(rows)
    if frame.empty:raise ValueError('No player team features')
    if not frame.groupby('game_id').size().eq(2).all():raise ValueError('Need paired team allocations')
    # Opponent projected production can describe the same game's environment.
    other=frame[['game_id','team','player_points','passing_yards','rushing_yards']].rename(columns={'team':'opponent','player_points':'opponent_player_points','passing_yards':'opponent_passing_yards','rushing_yards':'opponent_rushing_yards'})
    return frame.merge(other,on=['game_id','opponent'],validate='one_to_one')


def feature_columns(family):
    if family not in ['summary','components']:raise ValueError('Unknown two-way feature family')
    base=['baseline_score','is_home']+PLAYER_FEATURES+['opponent_player_points']
    return base+(COMPONENT_FEATURES if family=='components' else [])


def fit_player_score_correction(train,columns,ridge):
    allowed=set(feature_columns('components'))
    if not set(columns).issubset(allowed) or len(set(columns))!=len(columns):raise ValueError('Unsafe or duplicated score features')
    if ridge<=0 or not np.isfinite(ridge):raise ValueError('Positive finite ridge required')
    if train.empty or not np.isfinite(train.actual_score).all():raise ValueError('Complete finite training labels required')
    x=train[columns].to_numpy(float)
    if not np.isfinite(x).all():raise ValueError('Nonfinite player score features')
    center=x.mean(axis=0);scale=x.std(axis=0);scale=np.where(scale>1e-8,scale,1.)
    z=np.column_stack([np.ones(len(x)),(x-center)/scale])
    penalty=np.eye(z.shape[1])*ridge;penalty[0,0]=1e-8
    y=(train.actual_score-train.baseline_score).to_numpy(float)
    beta=np.linalg.solve(z.T@z+penalty,z.T@y)
    return dict(columns=list(columns),center=center.tolist(),scale=scale.tolist(),coefficients=beta.tolist(),ridge=float(ridge),training_team_games=len(train),
                feature_min=x.min(axis=0).tolist(),feature_max=x.max(axis=0).tolist(),support_policy='Observed training feature ranges; unsupported rows use the original team score')


def correction_support(frame,model):
    columns=model['columns']
    if not set(columns).issubset(set(feature_columns('components'))):raise ValueError('Unsafe score features')
    x=frame[columns].to_numpy(float)
    if not np.isfinite(x).all():raise ValueError('Nonfinite player score features')
    if 'feature_min' not in model or 'feature_max' not in model:raise ValueError('Training support bounds are required')
    lower=np.asarray(model['feature_min']);upper=np.asarray(model['feature_max'])
    return ((x>=lower-1e-9)&(x<=upper+1e-9)).all(axis=1)


def predict_player_score_correction(frame,model):
    supported=correction_support(frame,model)
    x=frame[model['columns']].to_numpy(float)
    z=np.column_stack([np.ones(len(x)),(x-np.asarray(model['center']))/np.asarray(model['scale'])])
    return np.where(supported,z@np.asarray(model['coefficients']),0.)


def two_way_projection(players,budgets,games,other_points,model,blend):
    if not 0<=blend<=1:raise ValueError('Invalid score correction blend')
    features=player_score_features(players,games,other_points)
    correction=predict_player_score_correction(features,model)
    scores=features[['game_id','team','baseline_score']].copy()
    scores['correction']=blend*correction
    scores['player_signal_supported']=correction_support(features,model)
    scores['score']=np.maximum(0.,scores.baseline_score+scores.correction)
    scores['other_points']=other_points
    result,updated,_=reconcile_scoring(players,budgets,scores[['game_id','team','score','other_points']])
    return result,updated,scores
