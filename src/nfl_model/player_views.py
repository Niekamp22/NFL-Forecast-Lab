"""Read-only matchup navigation and source-pinned player history."""
import hashlib
from pathlib import Path
import pandas as pd
from .artifact_paths import resolve_artifact

POSITION_STATS={
 'QB':['attempts','completions','passing_yards','passing_tds','passing_interceptions','carries','rushing_yards','rushing_tds'],
 'RB':['carries','rushing_yards','rushing_tds','targets','receptions','receiving_yards','receiving_tds'],
 'WR':['targets','receptions','receiving_yards','receiving_tds','carries','rushing_yards'],
 'TE':['targets','receptions','receiving_yards','receiving_tds'],
 'K':['fg_att','fg_made','pat_att','pat_made']}
PRIMARY={'QB':'passing_yards','RB':'rushing_yards','WR':'receiving_yards','TE':'receiving_yards','K':'fg_made'}

CARD_GROUPS={
 'QB':[('Passing',['attempts','completions','passing_yards','passing_tds','passing_interceptions']),('Rushing',['carries','rushing_yards','rushing_tds'])],
 'RB':[('Rushing',['carries','rushing_yards','rushing_tds']),('Receiving',['targets','receptions','receiving_yards','receiving_tds'])],
 'WR':[('Receiving',['targets','receptions','receiving_yards','receiving_tds']),('Rushing',['carries','rushing_yards'])],
 'TE':[('Receiving',['targets','receptions','receiving_yards','receiving_tds'])],
 'K':[('Field goals',['fg_att','fg_made']),('Extra points',['pat_att','pat_made'])]}
SHORT_LABELS={'passing_yards':'Yards','passing_tds':'TDs','passing_interceptions':'INTs',
 'rushing_yards':'Yards','rushing_tds':'TDs','receiving_yards':'Yards','receiving_tds':'TDs',
 'fg_att':'Attempts','fg_made':'Made','pat_att':'Attempts','pat_made':'Made'}


def with_total_yards(frame):
    """Display-only derived stat; both components must exist, including true zeros."""
    result=frame.copy()
    result['total_yards']=result.rushing_yards+result.receiving_yards
    return result


def season_hit_rates(history,player_id,season,stat,levels):
    """Descriptive frequencies from source-pinned completed games, not forecasts."""
    rows=with_total_yards(history[history.player_id.eq(player_id)&history.season.eq(season)])
    values=rows[stat].dropna()
    return pd.DataFrame([{'Milestone':f'{level:g}+','Hits':int(values.ge(level).sum()),
                          'Recorded games':len(values),'Season hit rate %':100*values.ge(level).mean() if len(values) else float('nan')}
                         for level in levels])


def blank_explanations(player,stats):
    """Explain only what the frozen row supports; don't infer injuries from blanks."""
    reasons={}
    for stat in stats:
        if pd.notna(player.get(stat,float('nan'))):continue
        if stat=='total_yards':
            reason='Requires both rushing and receiving yards; at least one is unavailable.'
        elif player.get('position')=='QB' and stat in ['attempts','completions','passing_yards','passing_tds','passing_interceptions'] and player.get('role')=='QB role unresolved':
            reason='No passing workload assigned to this QB; starter evidence is unresolved or another QB holds the likely-starter role.'
        elif player.get('history_games',0)==0:
            reason='No recent recorded usage on the current team to support this allocation.'
        else:
            reason='The saved forecast has no supported estimate for this stat; this does not mean zero production or an injury.'
        reasons.setdefault(reason,[]).append(stat.replace('_',' ').title())
    return reasons


def chronological_games(frame):
    result=frame.copy()
    result['local_kickoff']=pd.to_datetime(result.kickoff_utc,utc=True).dt.tz_convert('America/New_York')
    return result.sort_values(['local_kickoff','game_id']).reset_index(drop=True)


def matchup_players(players,game_id):
    return players[players.game_id.eq(game_id)&~players.is_unallocated].copy()


def load_pinned_history(meta):
    selected={resolve_artifact(p):h for p,h in meta['input_hashes'].items()
              if p.replace('\\','/').split('/')[-1].startswith('stats_player_week_') or p.replace('\\','/').split('/')[-1]=='games.parquet'}
    paths=list(selected)
    for p in paths:
        if hashlib.sha256(p.read_bytes()).hexdigest()!=selected[p]:
            raise ValueError(f'Historical source checksum mismatch: {p.name}')
    schedule=pd.read_parquet(next(p for p in paths if p.name=='games.parquet'))
    schedule=schedule[schedule.game_type.eq('REG')].copy()
    schedule['kickoff']=pd.to_datetime(schedule.gameday.astype(str)+' '+schedule.gametime.astype(str),errors='coerce').dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    stats=pd.concat([pd.read_parquet(p) for p in paths if p.name.startswith('stats_player_week_')],ignore_index=True)
    stats=stats[stats.season_type.eq('REG')]
    stats=stats.merge(schedule[['game_id','kickoff','home_score','away_score']],on='game_id',validate='many_to_one')
    return stats[stats.kickoff.lt(pd.Timestamp(meta['created_at']))&stats.home_score.notna()&stats.away_score.notna()]
