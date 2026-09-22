"""Small command-line entry point for data discovery and downloads."""
import argparse
import logging
from pathlib import Path

from .data.catalog import DATASETS
from .data.client import NFLVerseClient


def main() -> None:
    parser = argparse.ArgumentParser(description="nflverse data foundation")
    parser.add_argument("command", choices=["inspect", "download", "build", "report", "backfill", "backtest", "test-efficiency", "test-qb", "freeze-week", "grade-week", "qb-evidence", "test-totals", "freeze-scores", "test-probabilities", "forecast-week", "grade-forecast", "run-week", "project-players"])
    parser.add_argument("--season", type=int, nargs="+", default=[2026])
    parser.add_argument("--week", type=int, default=1)
    parser.add_argument("--team", type=str.upper)
    parser.add_argument("--output", type=Path, help="Save a report as Markdown")
    parser.add_argument("--dataset", choices=DATASETS, nargs="+")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--opportunity-model", action="store_true", help="Use experimental opportunity-based player projections")
    parser.add_argument("--defense-model", action="store_true", help="Build defense adjustment and historical matchup analysis from frozen opportunities")
    parser.add_argument("--split-matchup-model", action="store_true", help="Separate opponent workload and efficiency effects")
    parser.add_argument("--role-model", action="store_true", help="Freeze coherent role-based player projections")
    parser.add_argument("--defense-roles", action="store_true", help="Freeze defense-adjusted coherent role allocations")
    parser.add_argument("--milestones", action="store_true", help="Build and evaluate player milestone probabilities")
    parser.add_argument("--grade-players", action="store_true", help="Grade frozen role-based projections")
    parser.add_argument("--grade-displayed", action="store_true", help="Grade the exact current-policy snapshot displayed in the app; preserves original grading")
    parser.add_argument("--with-players", action="store_true", help="Include role forecasts and player grading in run-week")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.grade_displayed and not (args.command == 'project-players' and args.grade_players):
        parser.error('--grade-displayed requires project-players --grade-players')
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler("logs/ingestion.log")])
    client = NFLVerseClient(args.data_dir)
    if args.command == "project-players":
        if len(args.season) != 1:
            parser.error("project-players requires exactly one season")
        if args.milestones:
            from .models.milestones import build_milestones
            print(build_milestones(args.data_dir, args.season[0], args.week, defense=args.defense_roles))
        elif args.grade_players:
            from .models.player_grading import grade_role_week
            print(grade_role_week(client, args.season[0], args.week, args.force_refresh, defense=args.defense_roles, current=args.grade_displayed))
        elif args.defense_roles:
            from .models.role_defense import freeze_defense_roles
            print(freeze_defense_roles(args.data_dir, args.season[0], args.week))
        elif args.role_model:
            from .models.player_roles import freeze_role_forecast
            print(freeze_role_forecast(args.data_dir, args.season[0], args.week))
        elif args.split_matchup_model:
            from .models.player_matchup import run_player_matchup
            print(run_player_matchup(client, args.season[0], args.week))
        elif args.defense_model:
            from .models.player_defense import run_player_defense
            print(run_player_defense(client, args.season[0], args.week))
        else:
            from .models.player_projection import build_player_projections
            print(build_player_projections(client, args.season[0], args.week, args.opportunity_model))
    elif args.command == "run-week":
        if len(args.season) != 1:
            parser.error("run-week requires exactly one season")
        from .workflow import run_week
        import json
        result = run_week(client, args.season[0], args.week, args.force_refresh)
        if args.with_players:
            from .workflow import run_player_week
            result['players'] = run_player_week(client, args.season[0], args.week, args.force_refresh)
        print(json.dumps(result, indent=2))
    elif args.command == "test-probabilities":
        from .models.probability_experiment import run_probability_experiment
        print(run_probability_experiment(args.data_dir))
    elif args.command == "forecast-week":
        if len(args.season) != 1:
            parser.error("forecast-week requires exactly one season")
        from .models.weekly_forecast import freeze_weekly_forecast
        print(freeze_weekly_forecast(args.data_dir, args.season[0], args.week))
    elif args.command == "grade-forecast":
        if len(args.season) != 1:
            parser.error("grade-forecast requires exactly one season")
        from .models.full_grading import grade_full_week
        print(grade_full_week(client, args.season[0], args.week, args.force_refresh))
    elif args.command == "freeze-scores":
        if len(args.season) != 1:
            parser.error("freeze-scores requires exactly one season")
        from .models.score_forecast import freeze_scores
        print(freeze_scores(args.data_dir, args.season[0], args.week))
    elif args.command == "test-totals":
        from .models.totals_experiment import run_totals
        print(run_totals(args.data_dir))
    elif args.command == "qb-evidence":
        if len(args.season) != 1:
            parser.error("qb-evidence requires exactly one season")
        from .data.qb_evidence import build_qb_evidence
        print(build_qb_evidence(args.data_dir, args.season[0], args.week))
    elif args.command == "grade-week":
        if len(args.season) != 1:
            parser.error("grade-week requires exactly one season")
        from .models.grading import grade_week
        print(grade_week(client, args.season[0], args.week, args.force_refresh))
    elif args.command == "freeze-week":
        if len(args.season) != 1:
            parser.error("freeze-week requires exactly one season")
        from .models.prospective import freeze_week
        print(freeze_week(args.data_dir, args.season[0], args.week))
    elif args.command == "test-qb":
        from .models.qb_experiment import run_qb_experiment
        print(run_qb_experiment(args.data_dir))
    elif args.command == "test-efficiency":
        from .models.efficiency_experiment import run_efficiency_experiment
        print(run_efficiency_experiment(args.data_dir))
    elif args.command == "backtest":
        from .models.backtest import run_backtest
        print(run_backtest(args.data_dir))
    elif args.command == "backfill":
        from .data.history import backfill_history
        print(backfill_history(client, args.season, args.force_refresh))
    elif args.command == "report":
        if not args.team or len(args.season) != 1:
            parser.error("report requires --team and exactly one --season")
        from .report import team_report
        report = team_report(args.data_dir, args.season[0], args.week, args.team)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report, encoding="utf-8")
        print(report)
    elif args.command == "inspect":
        from .discovery import inspect_sources
        inspect_sources(client, args.season[0], args.week, args.force_refresh)
    elif args.command == "build":
        import json
        from .data.build import build_database
        result = build_database(client, args.season, args.force_refresh)
        print(json.dumps({"build_version": result["build_version"], **result["quality"]}, indent=2))
    else:
        for name in args.dataset or ["schedules", "play_by_play", "player_stats", "weekly_rosters"]:
            seasons = args.season if "{season}" in DATASETS[name].filename else None
            frame = client.load(name, seasons, force=args.force_refresh)
            print(f"{name}: {len(frame):,} rows, {len(frame.columns)} columns")
