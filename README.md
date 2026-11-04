# tradecalc

Fantasy basketball calculator for **head-to-head points leagues on ESPN, Sleeper and Yahoo**:
matchup odds, the league forecast, trade builder and finder, a streaming planner, player values and injuries.
Each league's own rules (scoring, lineup slots, weeks, ties, pickup limits) are read from its site. The default
league is ESPN league 1331882803; `tradecalc/config/league.yaml` holds the model settings and that league's ID.

## Use it for your league

Open the app, then **🏀 Open a league** in the sidebar:

| Site | What to enter | Private leagues |
|---|---|---|
| ESPN | the league ID, or paste your league page's link | add your `espn_s2` and `SWID` cookies (shown how in the app) |
| Sleeper | the league ID or link | none needed: Sleeper leagues are public |
| Yahoo | **Sign in with Yahoo**, then pick one of your leagues | Yahoo only shows a league to its members |
| (none) | **Try the demo league** | |

Pick your team and bookmark the page: the link (`?platform=…&league=…&team=…`) opens straight to it. Category and
roto leagues are recognized and explained, but not modeled yet.

**Privacy.**

- ESPN cookies and Yahoo sign-ins stay in your browser session; the app never saves them.
- League data fetched with them is cached on the server for up to an hour, under a hash of the login, so it's
  only ever shown to someone holding the same login.
- Public league data (team names, rosters, scores) is cached the same way for everyone.

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
| `tradecalc track [--excel FILE]` | Season tracker: record today's predictions and fill in finished weeks' results (`tracking/<season>/`) |
| `tradecalc injury-reports [--season S]` | Download and parse the NBA's official 5 PM injury report for every past game day (needs `pdfplumber`) |

## App

```bash
.venv/bin/streamlit run tradecalc/app.py
```

Tabs: Matchup (this week + rest of season), League forecast, Simulate (projection-engine Monte Carlo
for any two teams), Trade builder, Trade finder, Streaming (weekly drop / pickup plan for any team), Players, Injuries (with editable overrides and injury
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

## Development and releases

- **Branches.** Work happens on a branch (e.g. `dev/multi-league`) and is merged to `main` only when the tests
  pass. `main` is what Streamlit Cloud serves, and the daily tracker job also commits to it.
- **Tests.** `.venv/bin/pytest` runs the suite: model, platform adapters on recorded responses, streaming and
  trade rules. It takes seconds and needs no network.
  `TRADECALC_APP_SMOKE=1 .venv/bin/pytest tradecalc/tests/test_app_smoke.py` runs the whole app on the demo
  league with real data.
- **CI** (`.github/workflows/ci.yml`) runs the suite and the Discord bot's tests on every push. On pull requests
  and `main` it also runs the app smoke test.
- **Model changes** are backtested before they ship: `tradecalc audit`, with the results written up in
  `docs/model_audit.md`. The plan and its status are in `docs/production_plan.md`.

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
- If the default league goes private, add `ESPN_S2` and `SWID` under the app's *Settings → Secrets*. Visitors'
  own private leagues use their own cookies instead.
- **Yahoo sign-in** needs a free Yahoo developer app, set up once:
  1. Go to <https://developer.yahoo.com/apps/create/> and create a **Web Application**.
  2. Set the **Redirect URI** to the app's address (e.g. `https://your-app.streamlit.app/`), and give it
     **Fantasy Sports: Read** permission.
  3. Add `YAHOO_CLIENT_ID`, `YAHOO_CLIENT_SECRET` and `YAHOO_REDIRECT_URI` (the same address) to the app's
     secrets. Without them, the Yahoo option says it isn't set up.
- Refit the simulator every few weeks: `tradecalc sim-fit` (and `tradecalc sim-validate`), then commit and push
  `tradecalc/config/sim_params.yaml` and `validation.json`.

## Discord bot

Posts the standings (next to the record each team was supposed to have), the projected final standings and
the week's matchup odds in your league's Discord channel at the start of every fantasy week, and answers
`/odds team opponent`, `/standings`, `/forecast` and `/matchups` anytime, with the app's own numbers. The morning GitHub job refreshes its data (`tradecalc discord`); a free Cloudflare Worker answers
the commands. One-time setup: [`discord-bot/README.md`](discord-bot/README.md).

## Season tracker (prediction vs actual)

`.github/workflows/track.yml` runs `tradecalc track` every morning on GitHub Actions (9:17 AM ET,
before any tip-off) and commits `tracking/<season>/matchups.csv`, `players.csv` and `standings.csv`.
Predictions made before a week starts are frozen; results are filled in when the week ends. The app's
**📈 Season tracker** tab shows the standings next to the record each team was supposed to have (a win for every
week it was favored), every week that went against the odds, the running scorecard, and downloads everything as
one Excel workbook.
Run it by hand from the repo's Actions tab ("Run workflow") or locally with `tradecalc track`.

