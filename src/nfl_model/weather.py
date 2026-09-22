"""Display-only weather context. Never imported by projection code."""
import hashlib
from pathlib import Path
import pandas as pd
import requests
from .artifact_paths import resolve_artifact

FIELDS = ['temperature_2m', 'wind_speed_10m', 'wind_gusts_10m', 'precipitation_probability']


def venue_context(meta, game_id):
    source, expected = next((p, h) for p, h in meta['input_hashes'].items()
                            if p.replace('\\', '/').split('/')[-1] == 'games.parquet')
    path = resolve_artifact(source)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Schedule checksum mismatch')
    schedule = pd.read_parquet(path)
    game = schedule.set_index('game_id').loc[game_id]
    venues = pd.read_csv(Path(__file__).with_name('stadiums.csv')).set_index('stadium_id')
    context = dict(stadium=str(game.stadium), roof=str(game.roof).strip().lower())
    if game.stadium_id in venues.index:
        venue = venues.loc[game.stadium_id]
        if pd.notna(venue.lat) and pd.notna(venue.lon):
            context.update(latitude=float(venue.lat), longitude=float(venue.lon))
    return context


def kickoff_weather(latitude, longitude, kickoff, now=None):
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    target = pd.Timestamp(kickoff).tz_convert('UTC')
    if target <= now:
        return {'status': 'Game has started. No archived pregame weather forecast is available.'}
    if target.normalize() > now.normalize() + pd.Timedelta(days=15):
        return {'status': 'Kickoff is outside the weather forecast window. Check closer to game day.'}
    try:
        response = requests.get('https://api.open-meteo.com/v1/forecast', params={
            'latitude': latitude, 'longitude': longitude, 'hourly': ','.join(FIELDS),
            'temperature_unit': 'fahrenheit', 'wind_speed_unit': 'mph',
            'timezone': 'UTC', 'forecast_days': 16}, timeout=10)
        response.raise_for_status()
        hourly = response.json()['hourly']
        times = pd.to_datetime(hourly['time'], utc=True)
        index = abs(times - target).argmin()
        if abs(times[index] - target) > pd.Timedelta(minutes=30):
            return {'status': 'Forecast does not cover kickoff yet.'}
        values = {field: hourly[field][index] for field in FIELDS}
        return dict(status='ok', fetched_at=now.isoformat(), forecast_time=times[index].isoformat(), **values)
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return {'status': 'Weather provider is unavailable. Projections are still available.'}
