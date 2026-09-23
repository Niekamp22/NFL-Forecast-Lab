# NFL Forecast Lab

Read-only review app: 2026, Week 3. Matchups, player projections, selectable history charts, and experimental milestone probabilities.

## Deploy

Use a **private** GitHub repository. In Streamlit Community Cloud select branch `main`, entry point `streamlit_app.py`, and Python **3.13**. Configure the app for public viewing to share its app URL while keeping this repository private.

## Data and updates

Forecasts and player statistics are saved snapshots. Updating the code alone does not create new forecasts. Export a new reviewed bundle for a new slate. The matchup weather panel can request current Open-Meteo forecasts; weather never changes projections. No model training runs in the app.

Source: nflverse historical data. Forecast bytes and original manifests are preserved. `artifact-map.json` resolves original references on Linux; content checksums still apply. Only the matchup/player review flow is included. Local research and administrative pages remain in the development project.

Probabilities are experimental comparable-game estimates and have not been validated for the live role/defense forecast policy. They assume comparable participation. Missing estimates remain unknown.
