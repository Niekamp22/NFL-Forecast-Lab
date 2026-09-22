"""Weather has its own short-lived cache, separate from frozen forecasts."""
import json
import pandas as pd
import streamlit as st
from nfl_model.weather import venue_context, kickoff_weather


@st.cache_data(show_spinner=False)
def cached_venue(manifest, game_id):
    return venue_context(json.loads(manifest), game_id)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_weather(latitude, longitude, kickoff):
    return kickoff_weather(latitude, longitude, kickoff)


def render_weather(meta, game_id, kickoff):
    with st.expander('Game-day weather · information only', expanded=False):
        st.caption('Weather does not change player projections, team scores, or milestone probabilities.')
        try:
            venue = cached_venue(json.dumps(meta, sort_keys=True), game_id)
        except (OSError, ValueError, KeyError, StopIteration):
            st.info('Venue information unavailable.'); return
        roof = venue['roof']
        st.write(venue['stadium'])
        if roof in {'dome', 'closed'}:
            st.info('Schedule lists a covered or closed-roof game. Outdoor weather is not an on-field forecast.')
            return
        if roof != 'outdoors' and roof != 'open':
            st.caption('Roof status unconfirmed. Conditions below are outside the venue.')
        else:
            st.caption('Outdoor / open roof according to the saved schedule. Conditions are near the venue.')
        if 'latitude' not in venue:
            st.info('Venue coordinates unavailable.'); return
        if st.button('Refresh weather', key='refresh_weather'):
            cached_weather.clear()
        with st.spinner('Loading kickoff weather…'):
            weather = cached_weather(venue['latitude'], venue['longitude'], str(kickoff))
        if weather['status'] != 'ok':
            st.info(weather['status']); return
        for column, field, label, unit in zip(st.columns(4),
                ['temperature_2m', 'wind_speed_10m', 'wind_gusts_10m', 'precipitation_probability'],
                ['Temperature', 'Wind', 'Gusts', 'Precipitation chance'], ['°F', ' mph', ' mph', '%']):
            value = weather[field]
            column.metric(label, '—' if value is None else f'{value:.0f}{unit}')
        hour = pd.Timestamp(weather['forecast_time']).tz_convert('America/New_York').strftime('%b %d, %I:%M %p ET')
        fetched = pd.Timestamp(weather['fetched_at']).tz_convert('America/New_York').strftime('%b %d, %I:%M %p ET')
        st.caption(f'Nearest kickoff forecast hour: {hour} · Retrieved: {fetched} · Cached up to 30 minutes. Forecasts can change.')
        st.caption('Weather: [Open-Meteo](https://open-meteo.com/) · Venue coordinates: [Stadiums](https://github.com/greerreNFL/stadiums).')
