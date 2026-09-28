"""Observed scoring components and strictly prior-week matchup features."""
import numpy as np
import pandas as pd


def component_observations(pbp):
    if pbp.duplicated(['game_id','play_id']).any():
        raise ValueError('Duplicate play IDs')
    p=pbp[pbp.season_type.eq('REG') & pbp.posteam.notna() & pbp.defteam.notna()].copy()
    p['off_td']=(p.touchdown.eq(1)&p.td_team.eq(p.posteam)&~p.return_touchdown.eq(1)).astype(int)
    p['fg']=p.field_goal_result.eq('made').astype(int)
    p['turnovers']=(p.interception.eq(1)|p.fumble_lost.eq(1)).astype(int)
    events=p.groupby(['game_id','posteam','defteam'],as_index=False)[['off_td','fg','turnovers']].sum()
    meaningful=p[p.play_type.isin(['run','pass','field_goal','punt']) & ~p.qb_kneel.eq(1) & p.fixed_drive.notna()]
    drives=meaningful.groupby(['game_id','posteam','defteam','fixed_drive'],as_index=False).agg(yardline=('yardline_100','min'),td=('off_td','max'))
    drives['redzone']=drives.yardline.le(20).astype(int)
    drives['redzone_td']=drives.redzone*drives.td
    totals=drives.groupby(['game_id','posteam','defteam'],as_index=False).agg(drives=('fixed_drive','count'),redzone=('redzone','sum'),redzone_td=('redzone_td','sum'))
    return totals.merge(events,on=['game_id','posteam','defteam'],validate='one_to_one').rename(columns={'posteam':'team','defteam':'opponent'})


def prior_component_features(base,observations,window=10,prior_games=5):
    output=[]
    fields=['drives','off_td','fg','redzone','redzone_td','turnovers','other_points']
    for (season,week),targets in base.groupby(['season','week'],sort=True):
        cutoff=pd.to_datetime(targets.gameday).min()
        past=observations[((observations.season<season)|((observations.season==season)&(observations.week<week))) &
            (pd.to_datetime(observations.gameday)<cutoff)&(pd.to_datetime(observations.gameday)>=cutoff-pd.Timedelta(days=365))]
        if past.empty:raise ValueError('No prior component evidence')
        league=past[fields].mean()
        def estimate(team,defense=False):
            g=past[(past.opponent if defense else past.team).eq(team)].sort_values('gameday').tail(window)
            sums=g[fields].sum()+prior_games*league
            games=len(g)+prior_games
            return dict(drives=sums.drives/games,td_rate=sums.off_td/sums.drives,fg_rate=sums.fg/sums.drives,
                rz_rate=sums.redzone/sums.drives,rz_td_rate=sums.redzone_td/max(sums.redzone,1e-8),
                turnover_rate=sums.turnovers/sums.drives,other_points=sums.other_points/games)
        for r in targets.itertuples():
            for own,opp in [('home','away'),('away','home')]:
                row=dict(season=season,week=week,game_id=r.game_id,gameday=r.gameday,venue=own,is_home=float(own=='home'),
                    residual=getattr(r,own+'_score')-getattr(r,'projected_'+own+'_score'),
                    latest_feature_date=str(past.gameday.max()))
                for prefix,team,defense in [('off',getattr(r,own+'_team'),False),('def',getattr(r,opp+'_team'),True)]:
                    row.update({prefix+'_'+key:value for key,value in estimate(team,defense).items()})
                output.append(row)
    return pd.DataFrame(output)
