"""Read-only milestone panel for a selected player and statistic."""
import streamlit as st
from nfl_model.milestone_views import load_milestones
from nfl_model.player_views import load_pinned_history,season_hit_rates
from nfl_model.models.milestones import thresholds
import json


@st.cache_data(show_spinner=False)
def milestone_history(meta_json):
    return load_pinned_history(json.loads(meta_json))


def render_milestones(root,season,week,defense,folder,meta,player_id,game_id,position,stat):
    st.subheader('Historical milestone results')
    st.caption('175+ means at least 175, including exactly 175. Statistical milestones are not sportsbook settlement rules.')
    try:
        rates=season_hit_rates(milestone_history(json.dumps(meta,sort_keys=True)),player_id,season,stat,thresholds(stat))
        st.markdown(f'**{season}: what actually happened**')
        rates['Games reaching milestone']=rates.apply(lambda r:f"{int(r['Hits'])} of {int(r['Recorded games'])} games",axis=1)
        rates=rates[['Milestone','Games reaching milestone','Season hit rate %']] 
        st.dataframe(rates,hide_index=True,width='stretch',column_config={'Season hit rate %':st.column_config.NumberColumn('Season hit rate',format='%.1f%%')})
        st.caption('Hits / recorded games in this season across all teams, through the forecast’s saved data cutoff. Missing stats and unrecorded appearances are excluded. A 1/1 record is one observed result, not a 100% probability next game. These are stat hit rates, not model accuracy.')
    except (OSError,ValueError,KeyError,StopIteration) as exc:
        st.info(f'Season hit rates unavailable: {exc}')
    st.caption('Next-game milestone probabilities are not yet validated for the displayed projection. Historical results do not predict the next game.')
    with st.expander('Experimental research: comparable-game estimates'):
        render_research(root,season,week,defense,folder,meta,player_id,game_id,position,stat)


def render_research(root,season,week,defense,folder,meta,player_id,game_id,position,stat):
    try:
        data=load_milestones(root,season,week,defense,folder,meta)
        if data is None:
            st.info('Milestone probabilities have not been built for this frozen forecast yet.');return
        all_probs,evaluation,reliability,prob_meta=data
        probabilities=all_probs[all_probs.player_id.eq(player_id)&all_probs.game_id.eq(game_id)&all_probs.stat.eq(stat)].sort_values('threshold')
        available=probabilities[probabilities.status.eq('experimental')]
        if available.empty:
            reasons={'projection_unavailable':'The selected stat has no point projection.',
                     'limited_current_team_history':'Fewer than three recent games on the current team support this role allocation.',
                     'insufficient_training':'Too few historical comparable games are available.',
                     'outside_training_support':'This projection falls outside the historical training range; probabilities would require unsupported extrapolation.'}
            status=probabilities.status.iloc[0] if len(probabilities) else ''
            st.info('Probability unavailable: '+reasons.get(status,'No supported estimate for this player/stat.'));return
        st.warning('Experimental comparable-game estimates—not validated probabilities for this role/defense forecast. Historical tests use rolling player averages; today’s forecast uses a different allocation model. Estimates assume participation in a comparable role.')
        display=available[['threshold','probability','comparable_hits','comparable_games']].copy()
        display['Milestone']=display.threshold.map(lambda n:f'{n:g}+')
        display['Probability %']=100*display.probability
        display['Support']=display.apply(lambda r:'Sparse hit/miss evidence' if min(r.comparable_hits,r.comparable_games-r.comparable_hits)<10 else 'Comparable-game sample',axis=1)
        st.dataframe(display[['Milestone','Probability %','comparable_hits','comparable_games','Support']],hide_index=True,width='stretch',column_config={'Probability %':st.column_config.ProgressColumn('Research estimate',format='%.1f%%',min_value=0,max_value=100),'comparable_hits':'Historical hits','comparable_games':'Comparable games'})
        st.caption(f"Comparable pregame averages span {available.comparable_expected_low.iloc[0]:.1f}–{available.comparable_expected_high.iloc[0]:.1f}. These are player-games, not independent players. Sparse tail estimates are especially uncertain.")
        st.download_button('Download milestone estimates',available.to_csv(index=False),f'{player_id}_{stat}_milestones.csv','text/csv')
        with st.expander('Historical probability checks and method'):
            checked=evaluation[evaluation.position.eq(position)&evaluation.stat.eq(stat)]
            st.write('Training: 2022–2023. Evaluation: 2024–2025, already reused in research. Brier score measures probability error; lower is better. Frequency baseline ignores the pregame player expectation.')
            st.dataframe(checked,hide_index=True,width='stretch')
            bins=reliability[reliability.position.eq(position)&reliability.stat.eq(stat)]
            st.write('Reliability: compare mean probability with observed frequency. Bins pool related thresholds from the same games; counts are not independent samples.')
            st.dataframe(bins,hide_index=True,width='stretch')
            st.caption('Up to 300 closest historical pregame stat averages at the same position; half-count smoothing prevents unsupported certainty. No active probability is modeled. RB total yards uses observed rushing + receiving together. This is not a joint distribution across all stats.')
            st.caption(f"Probabilities frozen {prob_meta['created_at']}; matched to this forecast by checksum.")
    except (OSError,ValueError,KeyError) as exc:st.error(f'Milestone probabilities unavailable: {exc}')
