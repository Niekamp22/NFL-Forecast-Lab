"""Explain the exact team/player accounting of a saved forecast."""
import numpy as np
import pandas as pd


def team_accounting(players,game_id,team,score):
    group=players[players.game_id.eq(game_id)&players.team.eq(team)]
    if group.empty:raise ValueError('No forecast rows for this team and game')
    named=group[~group.is_unallocated]
    reserve=group[group.is_unallocated]
    if len(reserve)!=1:raise ValueError('Expected one team reserve')
    def total(rows,stat):
        return float(rows[stat].sum(min_count=1)) if len(rows) else 0.
    receiving=[]
    for label,rows in [('RB',named[named.position.eq('RB')]),('WR',named[named.position.eq('WR')]),
                       ('TE',named[named.position.eq('TE')]),('QB / other',named[~named.position.isin(['RB','WR','TE'])]),
                       ('Unassigned receiving yards',reserve)]:
        receiving.append({'Allocation':label,'Receiving yards':total(rows,'receiving_yards')})
    passing=total(group,'passing_yards');caught=total(group,'receiving_yards')
    if not np.isclose(passing,caught,atol=1e-7):raise ValueError('Saved passing and receiving totals do not reconcile')
    td=6*(total(group,'receiving_tds')+total(group,'rushing_tds'))
    fg=3*total(group,'fg_made');pat=total(group,'pat_made')
    joint='scoring_team_points' in group and group.scoring_team_points.notna().all()
    other=float(group.scoring_other_points.iloc[0]) if joint else None
    if joint and not np.isclose(td+fg+pat+other,score,atol=1e-7):raise ValueError('Shared scoring and displayed score differ')
    return dict(receiving=pd.DataFrame(receiving),passing_yards=passing,receiving_yards=caught,
                named_passing_yards=total(named,'passing_yards'),unassigned_passing_yards=total(reserve,'passing_yards'),
                unassigned_receiving_yards=total(reserve,'receiving_yards'),
                touchdown_points=td,field_goal_points=fg,extra_point_points=pat,
                player_points=td+fg+pat,team_score=float(score),score_difference=float(score)-td-fg-pat,joint_scoring=joint,other_points=other)
