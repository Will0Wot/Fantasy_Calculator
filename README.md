# tradecalc

Fantasy basketball trade calculator for ESPN league 1331882803 (H2H points, 24 weeks, no playoffs).
Every league setting lives in `tradecalc/config/league.yaml`.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

| Command | What it does |
|---|---|
| `tradecalc check` | Verify ESPN, schedule, box scores and ADP; print week boundaries |
| `tradecalc refresh` | Pull new games, injuries and rosters |
| `tradecalc values [--team ID] [--top 50]` | Rest-of-season values and flags |
| `tradecalc evaluate "A + B for C" [--with ID] [--precise]` | One trade, both teams |
| `tradecalc find [--with ID] [--max-size 2] [--top 10]` | Trades that help you and the partner might accept |
| `tradecalc week [--week N] [--precise]` | This week's projected matchup and win probability |
| `tradecalc lineup [--week N] [--team ID]` | Simulated starts per player |
| `tradecalc injuries [--all]` | Injured / recently returned players and what the model assumes |
| `tradecalc backtest` | Model error table; saves the chosen `k` |
| `tradecalc sim-fit` | Fit the projection engine on the two latest seasons |
| `tradecalc simulate [--a ID] [--b ID] [--week N] [--sims 10000]` | Monte Carlo of any two fantasy teams |
| `tradecalc sim-validate` | Out-of-sample validation of the engine on last season |
| `tradecalc forecast-test` | Backtest the league forecast on mock leagues from last season |
| `tradecalc audit [--baseline] [--quick]` | Full model audit on 2023-24, 2024-25, 2025-26 + break tests (Model audit tab) |

## App

```bash
.venv/bin/streamlit run tradecalc/app.py
```

Tabs: Matchup (this week + rest of season), League forecast, Simulate (projection-engine Monte Carlo
for any two teams), Trade builder, Trade finder, Players, Injuries (with editable overrides and injury
history), Forecast tests, Model audit (multi-season accuracy tests, before/after, break tests;
report in `docs/model_audit.md`), Methodology (every formula: `docs/methodology.md`), Help.

## Projection engine
Game-by-game simulation with per-stat models, minutes from rotations/depth charts, teammate usage,
opponent defense and pace, injuries, back-to-back rest and coach tendencies. Method and validation:
`docs/projection_engine.md`.

All commands accept `--today YYYY-MM-DD` to pretend it is another day.

## Data
- ESPN public API: league settings, rosters, matchup schedule, NBA schedule, injuries, ADP, projections.
  If the league goes private again, put `ESPN_S2` and `SWID` in `.env` (see `.env.example`).
- stats.nba.com via `nba_api`: box scores, cached in `tradecalc/data/cache.sqlite`.
- `tradecalc/data/adp.csv`, `adp_prev.csv`, `adp_2025.csv`: ADP (or ESPN draft rank) CSVs; columns `name,adp`.
  Replace the historical ones with genuine ADP (e.g. FantasyPros) for a cleaner ADP curve.
- ESPN's public NBA injury report: current injuries with expected return dates (logged each refresh).
- `tradecalc/data/injury_overrides.csv`: `name,status,games_out,return_date` to override a current injury.
- `tradecalc/data/injury_history.csv`: `name,injury,returned` to add a past injury.

## Injuries
Each injury is classified into a research-based profile (`tradecalc/config/injury_profiles.yaml`)
that sets return timing, P(returns this season), and year-1/year-2 effects on production,
games played and volatility. Research, sources and derivations: `docs/injury_research.md`.

## Things to check
- Week boundaries are inferred (week 1 = Oct 20-25, All-Star week 18 = Feb 15-28). If ESPN's schedule
  page differs, set `weeks.overrides` in league.yaml.

## Tests
```bash
.venv/bin/pytest
```

## Deploying for the league (Streamlit Community Cloud)

1. Push this folder to a GitHub repository (private is fine).
2. On share.streamlit.io: **Create app** → pick the repository, branch `main`, main file `tradecalc/app.py`,
   and Python 3.12 under *Advanced settings*. Deploy.
3. Share the app link. Each person picks their team in the sidebar; the link remembers it (`?team=<id>`).

Notes
- The local cache (`tradecalc/data/cache.sqlite`) is not committed. Finished seasons' NBA data ships as small
  compressed files in `tradecalc/data/seed/` (box scores, team logs, ages/usage, on/off), loaded into the cache on
  first start, because stats.nba.com often blocks cloud servers. Everything else (ESPN rosters, injuries, depth
  charts, this season's games) is downloaded live. This season's box scores fall back to ESPN automatically when
  stats.nba.com is unreachable. After a season ends: `tradecalc seed-export`, then commit the new seed files.
- "Today" is the league's clock (US Eastern, `league.timezone`), not the server's, which is UTC on Streamlit Cloud.
- League data is rebuilt at least hourly (cache expiry), or right away with **Refresh data**.
- On the hosted copy, injury corrections are read-only. To change them, edit
  `tradecalc/data/injury_overrides.csv` / `injury_history.csv` locally, commit and push; the app redeploys.
- If the league goes private, add `ESPN_S2` and `SWID` under the app's *Settings → Secrets*.
- Refit the simulator every few weeks: `tradecalc sim-fit` (and `tradecalc sim-validate`), then commit and push
  `tradecalc/config/sim_params.yaml` and `validation.json`.
