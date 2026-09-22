import streamlit as st
import pandas as pd
from nfl_model.player_views import with_total_yards, POSITION_STATS


def player_browser(players, go):
    st.subheader('Find a player')
    rows = with_total_yards(players[~players.is_unallocated]).copy()
    a, b, c = st.columns([2, 1, 1])
    query = a.text_input('Player name', placeholder='Search any player in this week')
    team = b.selectbox('Team', ['All teams'] + sorted(rows.team.unique()))
    position = c.selectbox('Position', ['All positions'] + list(POSITION_STATS))
    rows = rows[rows.player_name.fillna('').str.contains(query, case=False, regex=False)]
    if team != 'All teams': rows = rows[rows.team.eq(team)]
    if position != 'All positions': rows = rows[rows.position.eq(position)]
    rows = rows.sort_values(['player_name','team']).reset_index(drop=True)
    if rows.empty:
        st.info('No players match these filters.'); return
    rows['selection_id'] = rows.game_id + ':' + rows.player_id
    indexed = rows.set_index('selection_id')
    selected = st.selectbox('Open player breakdown', rows.selection_id.tolist(), index=None,
        placeholder='Type or choose a player…',
        format_func=lambda i: f'{indexed.loc[i].player_name} · {indexed.loc[i].team} · {indexed.loc[i].position}')
    if selected is not None:
        r = indexed.loc[selected]
        st.button('View selected player →', on_click=go, args=(r.game_id,r.player_id))
    with st.expander('Weekly player overview', expanded=bool(query) or team!='All teams' or position!='All positions'):
        fields = POSITION_STATS[position] if position!='All positions' else ['passing_yards','rushing_yards','receiving_yards','receptions']
        if position=='RB': fields = fields + ['total_yards']
        display = rows[['player_name','team','opponent','position']+fields].rename(columns=lambda x:x.replace('_',' ').title())
        st.caption('Click a column header to sort. Blank means no supported estimate. These are saved projections, not live statistics.')
        st.dataframe(display, hide_index=True, width='stretch')
        comparison = st.multiselect('Compare up to three players', rows.selection_id.tolist(), max_selections=3,
            format_func=lambda i: f'{indexed.loc[i].player_name} · {indexed.loc[i].team}')
        if comparison:
            selected_rows = indexed.loc[comparison]
            comparison_table = selected_rows[['player_name','team','position']+fields].copy()
            comparison_table.index = [f'{r.player_name} ({r.team})' for r in selected_rows.itertuples()]
            comparison_table = comparison_table.drop(columns=['player_name','team']).T
            comparison_table.index = comparison_table.index.str.replace('_',' ').str.title()
            st.dataframe(comparison_table.map(lambda v: '—' if pd.isna(v) else str(v) if isinstance(v,str) else f'{v:.1f}'), width='stretch')
        st.download_button('Download filtered overview', display.to_csv(index=False), 'weekly_player_overview.csv','text/csv')


def results_panel(result, game_id=None, player_id=None):
    with st.expander('Saved projections vs actual results'):
        st.caption('Only results matched to the exact displayed forecast version are shown. A saved forecast is not proof of when it was published on the website.')
        if result is None:
            st.info('No verified results match this forecast version yet. Older forecast versions are not substituted.'); return
        _, run, rows = result
        if game_id: rows = rows[rows.game_id.eq(game_id)]
        if player_id: rows = rows[rows.player_id.eq(player_id)]
        rows = rows.loc[rows.apply(lambda r: r.stat in POSITION_STATS.get(r.position,[]), axis=1).astype(bool)]
        graded = rows[rows.status.eq('graded')]
        st.caption(f"Results checked: {run['created_at']} · {len(graded)} graded player-stat observations. Missing results are excluded, never treated as zero.")
        if graded.empty:
            st.info('No verified player results available for this selection yet.'); return
        st.dataframe(graded[['player_name','team','position','stat','projection','actual','error']].rename(columns={'error':'Projection minus actual'}),hide_index=True,width='stretch')
        summary = graded.groupby(['position','stat']).error.agg(Observations='size',Average_miss=lambda s:s.abs().mean()).reset_index()
        st.dataframe(summary,hide_index=True,width='stretch')
