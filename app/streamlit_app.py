"""Matchups → game → player, backed by immutable local forecasts."""
import json
import os
import sys
from pathlib import Path
import pandas as pd
import altair as alt
import streamlit as st

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
sys.path.insert(0,str(PROJECT/'app'))
from milestone_panel import render_milestones
from weather_panel import render_weather
from consumer_panel import player_browser, results_panel
from nfl_model.consumer_views import matching_player_results, game_status, date_label
from nfl_model.dashboard import available_weeks,load_forecast
from nfl_model.models.player_roles import checked_player_snapshot
from nfl_model.player_views import POSITION_STATS,PRIMARY,CARD_GROUPS,SHORT_LABELS,chronological_games,matchup_players,load_pinned_history,with_total_yards,blank_explanations

st.set_page_config(page_title='NFL Forecast Lab',page_icon='🏈',layout='wide')
st.markdown('''<style>.block-container{padding-top:4rem;max-width:1360px}
h1{letter-spacing:-.045em} [data-testid="stMetric"]{padding:8px 0}
[data-testid="stVerticalBlockBorderWrapper"]{border-radius:14px}
button[kind="primary"]{border-radius:8px}</style>''',unsafe_allow_html=True)
root=Path(os.environ.get('NFL_MODEL_DATA_DIR',str(PROJECT/'data'))).resolve()


def go(game=None,player=None):
    st.query_params['season']=str(season);st.query_params['week']=str(week)
    st.query_params['model']='defense' if use_defense_roles else 'baseline'
    for name,value in [('game',game),('player',player)]:
        if value is None:
            if name in st.query_params:del st.query_params[name]
        else:st.query_params[name]=value


def fmt(value):return '—' if pd.isna(value) else f'{value:.1f}'


@st.cache_data(show_spinner=False)
def historical_data(manifest_json):return load_pinned_history(json.loads(manifest_json))


st.sidebar.header('NFL Forecast Lab')
try:
    weeks=available_weeks(root)
    if not weeks:st.info('No frozen forecasts yet. Run the weekly pipeline to prepare a slate.');st.stop()
    seasons=sorted({s for s,w in weeks},reverse=True)
    requested=st.query_params.get('season','')
    season=st.sidebar.selectbox('Season',seasons,index=seasons.index(int(requested)) if requested.isdigit() and int(requested) in seasons else 0)
    choices=[w for s,w in weeks if s==season];requested=st.query_params.get('week','')
    week=st.sidebar.selectbox('Week',choices,index=choices.index(int(requested)) if requested.isdigit() and int(requested) in choices else 0)
    model_label=st.sidebar.selectbox('Player forecast',['Recent usage estimate','Defense-adjusted estimate (experimental)'],index=1 if st.query_params.get('model')=='defense' else 0)
    use_defense_roles=model_label.startswith('Defense')
    st.query_params['model']='defense' if use_defense_roles else 'baseline'
    scope=(season,week)
    if st.session_state.get('scope',scope)!=scope:go()
    st.session_state['scope']=scope
    _,team_meta,games=load_forecast(root,season,week)
    games=chronological_games(games)
    role_root=root/('player_role_defense_forecasts' if use_defense_roles else 'player_role_forecasts')/str(season)/f'week_{week:02d}'
    players=pd.DataFrame();budgets=pd.DataFrame();meta=None
    if list(role_root.glob('*/manifest.json')):
        folder,meta,players=checked_player_snapshot(role_root,current=True)
        budgets=pd.read_parquet(folder/'team_budgets.parquet')
except (OSError,ValueError,KeyError) as exc:st.error(f'Cannot load verified forecasts: {exc}');st.stop()

st.sidebar.button('All matchups',on_click=go,width='stretch')
if st.sidebar.button('Reload saved data',width='stretch'):
    historical_data.clear();st.rerun()
if os.environ.get('NFL_REVIEW_BUILD')!='1':
    st.sidebar.page_link('pages/3_Scorecards.py',label='Baseline results & team totals',icon='📋')
    st.sidebar.page_link('pages/2_Model_Research.py',label='Model research',icon='🔬')
else:
    st.sidebar.info('Preview for feedback · Saved forecasts, not a live feed. Player projections and milestone probabilities are experimental.')
st.sidebar.caption('Saved estimates · Reload reads saved files; it does not fetch new stats or injuries.')
if use_defense_roles:
    st.sidebar.caption('Defense-adjusted player workloads and yardage. Team scores and kickers unchanged. Historical comparison is at team level; player roles remain unvalidated.')
    if meta is not None:
        with st.sidebar.expander('Defense model comparison'):
            st.dataframe(pd.read_csv(folder/'evaluation.csv'),hide_index=True)
            st.caption('2024–2025 reused evaluation. Lower MAE is better. These are team-yardage errors, not player errors.')
results=None
if meta is not None:
    try:
        results=matching_player_results(root,season,week,use_defense_roles,meta)
    except (OSError,ValueError,KeyError):
        st.warning('Verified results could not be loaded for this forecast.')
    try:
        source_history=historical_data(json.dumps(meta,sort_keys=True))
        input_date=date_label(source_history.kickoff.max())
    except (OSError,ValueError,KeyError,StopIteration):
        input_date='Unavailable'
def render_freshness():
    st.caption('Saved forecast · No live injury updates')
    with st.expander('Forecast dates & data coverage'):
        st.write('Team estimates saved: '+date_label(team_meta.get('created_at')))
        if meta is not None:
            st.write('Player estimates saved: '+date_label(meta.get('created_at')))
            st.write('Latest recorded game in player inputs: '+input_date)
        st.caption('Dates describe the saved inputs, not a live feed. A game starting does not automatically update its results here.')
final_games=set() if results is None else set(results[2].loc[results[2].status.eq('graded'),'game_id'])

game_id=st.query_params.get('game');player_id=st.query_params.get('player')
if game_id and game_id not in set(games.game_id):
    st.warning('That matchup is not in this slate. Showing all matchups.');game_id=None;player_id=None

if not game_id:
    st.title('Matchups & player estimates')
    render_freshness()
    st.caption(f'{season} · WEEK {week} · {len(games)} GAMES · ALL TIMES EASTERN')
    st.write('Choose a matchup to explore both teams and their player projections.')
    if not players.empty:
        with st.expander('Search players & compare weekly estimates'):
            player_browser(players,go)
    results_panel(results)
    game_filter=st.radio('Show games',['All games','Upcoming','Started / final'],horizontal=True)
    visible_games=games.copy()
    started=visible_games.local_kickoff.le(pd.Timestamp.now(tz='UTC'))
    if game_filter=='Upcoming': visible_games=visible_games[~started]
    elif game_filter=='Started / final': visible_games=visible_games[started]
    if visible_games.empty: st.info('No games in this view.')
    for day,day_games in visible_games.groupby(games.local_kickoff.dt.date,sort=True):
        st.subheader(pd.Timestamp(day).strftime('%A, %B %d'))
        records=list(day_games.itertuples())
        for start in range(0,len(records),2):
            for column,game in zip(st.columns(2),records[start:start+2]):
                with column.container(border=True):
                    st.caption(game.local_kickoff.strftime('%I:%M %p ET')+' · '+game_status(game.local_kickoff,game.game_id in final_games))
                    st.subheader(f'{game.away_team} at {game.home_team}')
                    a,b=st.columns(2)
                    a.metric(f'{game.away_team} projected',fmt(game.projected_away_score))
                    b.metric(f'{game.home_team} projected',fmt(game.projected_home_score))
                    st.caption(f'Projected total {game.score_total:.1f} · {game.home_team} win {game.p_home:.0%}')
                    if not budgets.empty:
                        unresolved=budgets[budgets.team.isin([game.away_team,game.home_team])&~budgets.qb_resolved]
                        if len(unresolved):st.caption('QB role review: '+', '.join(unresolved.team))
                    st.button('View matchup →',key=f'game_{game.game_id}',on_click=go,args=(game.game_id,),width='stretch',type='primary')
    st.caption('Scores and probabilities are from the frozen team model. They do not account for unresolved QB changes.')
    st.stop()

game=games.set_index('game_id').loc[game_id]
label=f'{game.away_team} at {game.home_team}'
b1,b2=st.columns([1,4])
b1.button('← Matchups',on_click=go,width='stretch')
if player_id:b2.button(f'{label} → Players',on_click=go,args=(game_id,),key='breadcrumb_game')
else:b2.caption(f'Matchups / {label}')
if players.empty:
    st.title(label);st.info('Player projections have not been frozen for this slate yet.');st.stop()
roster=with_total_yards(matchup_players(players,game_id))
if player_id and player_id not in set(roster.player_id):
    st.warning('That player is not in this frozen matchup. Choose a player below.');player_id=None

if not player_id:
    st.title(label)
    render_freshness()
    st.caption(game.local_kickoff.strftime('%A, %B %d · %I:%M %p ET')+' · '+game_status(game.local_kickoff,game_id in final_games))
    a,b,c=st.columns(3)
    a.metric(f'{game.away_team} projected score',fmt(game.projected_away_score))
    b.metric(f'{game.home_team} projected score',fmt(game.projected_home_score))
    c.metric('Projected total',fmt(game.score_total))
    st.caption('Choose a position, then click a player for their full breakdown. Blank estimates remain unknown; role projections are provisional.')
    st.caption('Estimate method: '+model_label)
    if meta is not None: render_weather(meta,game_id,game.local_kickoff)
    results_panel(results,game_id)
    for tab,position in zip(st.tabs(list(POSITION_STATS)),POSITION_STATS):
        with tab:
            for column,team in zip(st.columns(2),[game.away_team,game.home_team]):
                with column:
                    st.subheader(team)
                    subset=roster[roster.team.eq(team)&roster.position.eq(position)].sort_values(PRIMARY[position],ascending=False)
                    if subset.empty:st.info('No eligible players recorded for this position.')
                    for row in subset.itertuples():
                        with st.container(border=True):
                            st.button(str(row.player_name)+' →',key=f'player_{row.player_id}',on_click=go,args=(game_id,row.player_id),width='stretch')
                            if position=='RB':
                                st.metric('Total yards',fmt(row.total_yards),help='Rushing yards + receiving yards. Both projections are required; excludes return yards.')
                            for group,fields in CARD_GROUPS[position]:
                                st.markdown(f'**{group}**')
                                st.caption(' · '.join(f'{SHORT_LABELS.get(s,s.title())}: {fmt(getattr(row,s))}' for s in fields))
                            reasons=blank_explanations(row._asdict(),POSITION_STATS[position]+(['total_yards'] if position=='RB' else []))
                            if reasons:
                                with st.expander('Why are some stats blank?'):
                                    for reason,fields in reasons.items():st.write(f"{', '.join(fields)}: {reason}")
                            if row.role=='QB role unresolved' or row.history_games==0:st.caption('⚠ '+row.role+' · '+str(row.history_games)+' recent team games')
                            elif 'Injury: ' in row.review_flags:st.caption('⚠ Availability review needed')
    with st.expander('Workload not assigned to a player & matchup notes'):
        reserve=players[players.game_id.eq(game_id)&players.is_unallocated]
        st.dataframe(reserve[['team','attempts','targets','carries']],hide_index=True,width='stretch')
        st.write(f'{game.away_team}: {game.away_qb_review}')
        st.write(f'{game.home_team}: {game.home_qb_review}')
        st.caption('Reserved shares are not assigned to a guessed starter. Team scores and player stat totals are separate models.')
    st.download_button('Download matchup projections',roster.to_csv(index=False),f'{game_id}_players.csv','text/csv')
    st.stop()

player=roster.set_index('player_id').loc[player_id]
st.caption(f'Matchups / {label} / {player.player_name}')
st.title(str(player.player_name))
render_freshness()
st.caption(f'{player.team} · {player.position} · vs {player.opponent} · Week {week}')
st.caption('Estimate method: '+model_label)
st.caption(game_status(game.local_kickoff,game_id in final_games))
st.info('Why this estimate? Recent playing opportunities and production set the starting point. '+('The opponent’s recent defensive results adjust workload and yardage.' if use_defense_roles else 'Opponent strength is not included in this selected estimate.')+' Weather is informational only; injuries are not updated live.')
st.subheader('Projected stat line')
stats=POSITION_STATS[player.position]+(['total_yards'] if player.position=='RB' else [])
stat_key=f'stat_{season}_{week}_{game_id}_{player_id}'
if st.session_state.get(stat_key) not in stats:st.session_state[stat_key]=PRIMARY[player.position]
def select_stat(stat):st.session_state[stat_key]=stat
for start in range(0,len(stats),4):
    for col,stat in zip(st.columns(4),stats[start:start+4]):
        col.button(stat.replace('_',' ').title(),key='explore_'+stat,on_click=select_stat,args=(stat,),width='stretch',type='primary' if st.session_state[stat_key]==stat else 'secondary')
        col.metric(stat.replace('_',' ').title(),fmt(player[stat]),label_visibility='collapsed')
st.caption('— means unavailable, not zero. Values are rounded; 0.0 may be a small estimate. Fractional touchdowns and kicks are expected counts, not scoring probabilities.')
if player.position=='RB':st.caption('Total yards = rushing + receiving yards, excluding return yards. It remains blank if either component is unavailable.')
reasons=blank_explanations(player,stats)
if reasons:
    with st.expander('Why are some stats blank?',expanded=True):
        for reason,fields in reasons.items():st.write(f"{', '.join(fields)}: {reason}")
selected_stat=st.selectbox('Explore a stat',stats,key=stat_key,format_func=lambda s:s.replace('_',' ').title())
st.subheader(selected_stat.replace('_',' ').title()+' · History & milestones')
try:
    history=historical_data(json.dumps(meta,sort_keys=True))
    past=with_total_yards(history[history.player_id.eq(player_id)].sort_values('kickoff').tail(10))
    if past.empty:st.info('No recorded historical games available for this player.')
    else:
        st.caption('Last 10 recorded regular-season games from the forecast’s saved sources. Includes previous teams and partial games; missing game rows are not inserted as zero.')
        chart=past[['season','week',selected_stat]].copy()
        chart['Game']=chart.season.astype(str)+' W'+chart.week.astype(str)
        chart=chart.rename(columns={selected_stat:'Actual'})
        bars=alt.Chart(chart).mark_bar(color='#2684c7').encode(x=alt.X('Game:N',sort=chart.Game.tolist(),title='Recorded game'),y=alt.Y('Actual:Q',title=selected_stat.replace('_',' ').title()),tooltip=['Game','Actual'])
        if pd.notna(player[selected_stat]):
            line=alt.Chart(pd.DataFrame({'Projection':[float(player[selected_stat])]})).mark_rule(color='#ef8a28',strokeDash=[6,4],strokeWidth=2).encode(y='Projection:Q',tooltip=['Projection'])
            bars=bars+line
        st.altair_chart(bars,width='stretch')
        st.caption('Blue bars: recorded results. Orange dashed line: the current selected forecast, not a forecast made for those historical games.')
        with st.expander('Recent game stat table'):st.dataframe(past[['season','week','team','opponent_team']+stats].iloc[::-1],hide_index=True,width='stretch')
except (OSError,ValueError,KeyError,StopIteration) as exc:st.warning(f'Historical detail unavailable: {exc}')
render_milestones(root,season,week,use_defense_roles,folder,meta,player_id,game_id,player.position,selected_stat)
with st.expander('Role & availability'):
    st.write(player.role);st.info(player.review_flags)
    a,b=st.columns(2)
    a.metric('Recent games on current team',int(player.history_games))
    b.metric('Prior snap share','Unknown' if pd.isna(player.expected_snap_share) else f'{player.expected_snap_share:.0%}')
    st.caption('Prior snap share is observed usage, not a prediction of participation. Starter designations are unconfirmed.')
results_panel(results,game_id,player_id)
with st.expander('How this projection was calculated'):
    team_budget=budgets[budgets.team.eq(player.team)].iloc[0]
    st.write('Team workloads use five recent team games. Player target and carry shares use up to five actual appearances in their current team stint, weighted toward newer games. One appearance gets full weight; unplayed games are not zeros. Shares are scaled to stay within the team budget; unresolved shares remain unallocated.' if meta.get('role_policy') in ['observed_current_stint_v2','current_season_v3','current_season_qb_v4'] else 'Team workloads use five recent team games, weighted toward newer games. Target and carry shares reflect observed usage on the current team; unresolved shares remain unallocated.')
    if meta.get('season_weighting'):
        st.caption(f"For workload shares, each {season} appearance gets {meta['season_weighting']['player_share']}× the weight of an equally recent older appearance. Older games still stabilize small samples. Team-volume, efficiency, and defensive weighting remain unchanged after historical checks.")
    if use_defense_roles:
        st.write(f"Opponent adjustment: passing workload × {player.pass_volume_multiplier:.3f}, receiving yards per target × {player.pass_rate_multiplier:.3f}; rushing workload × {player.rush_volume_multiplier:.3f}, yards per carry × {player.rush_rate_multiplier:.3f}.")
        st.caption(player.defense_status)
        compare=pd.DataFrame([{'Stat':s.replace('_',' ').title(),'Role baseline':player.get('baseline_'+s),'Defense adjusted':player[s]} for s in stats if s!='total_yards'])
        st.dataframe(compare,hide_index=True,width='stretch')
        st.caption('Shares remain fixed. TDs, completions and interceptions scale with workload only; no direct rate adjustment for those stats. Kicker estimates are unchanged.')
    for workload,production in [('targets','receiving_yards'),('carries','rushing_yards')]:
        if workload in stats and pd.notna(player[workload]) and player[workload]>0 and pd.notna(player[production]):
            rate=player[production]/player[workload]
            share=player[workload]/team_budget[workload] if team_budget[workload]>0 else 0
            st.write(f'{player[workload]:.1f} {workload} ({share:.1%} of team budget) × {rate:.2f} yards per opportunity = {player[production]:.1f} {production.replace("_"," ")}.')
    if player.position=='QB':st.write('Passing workload is assigned when the saved depth-chart QB1 matches the latest game’s majority passer, with at least 10 attempts. This is an unconfirmed role estimate, not live injury clearance. Passing yards, completions, and TDs equal the combined receiving allocations, including reserved production. An unresolved QB receives no passing estimate.' if meta.get('role_policy')=='current_season_qb_v4' else 'Passing workload is assigned only when depth rank and recent attempt leaders agree. Passing yards, completions, and TDs equal the combined receiving allocations, including reserved production. An unresolved QB receives no passing estimate.')
    if player.position=='K':st.write('Field-goal and extra-point attempts are allocated within recent team budgets; projected makes use historical conversion rates. Unknown shares stay reserved.')
    st.caption('Efficiencies use up to 10 prior games. Displayed multiplication may differ slightly due to rounding. No calibrated uncertainty intervals are available.'+(' Opponent multipliers are included in this selected version.' if use_defense_roles else ' This baseline has no opponent adjustment.'))
with st.expander('Matchup context & source timing'):
    st.write(f'{game.away_team}: {game.away_qb_review}')
    st.write(f'{game.home_team}: {game.home_qb_review}')
    st.caption(f"Player forecast frozen {meta['created_at']}. Historical sources are pinned; this is not a live injury feed. Defensive experiments remain on Model Research.")
st.download_button('Download player projection',roster[roster.player_id.eq(player_id)].to_csv(index=False),f'{player_id}_projection.csv','text/csv')
