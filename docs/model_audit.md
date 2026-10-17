# Model audit report

*Scope: the season model (Matchup tab, Trade builder, Trade finder, quick league forecast), the
projection engine (Simulate tab, detailed league forecast), the availability/injury layer, and the
backtest harness itself. Every formula is in `docs/methodology.md`; the raw results are in
`tradecalc/config/audit.json` (final model) and `audit_baseline.json` (original model), and are
re-created with `tradecalc audit --baseline`.*

## 1. Summary

- **What works.**
  - Player point projections beat a season-average baseline in every test season.
  - The simulator's ranges are honest.
  - The simulator's weekly win probabilities are calibrated.
  - The trade calculator's accounting holds: rosters, drops, free agents, lineups and zero-sum. Every break test passes.
- **What didn't, and is fixed.**
  - The trade calculator's weekly win chances were badly overconfident: a calibration slope of 0.39–0.49, where 1.0 is honest.
  - Its trade gains were overstated: predicted gains for ACCEPT trades were 1.19–1.40 wins, against 0.81–1.07 real.
  - The causes:
    - Raw games-played rates were used as probabilities.
    - Two sources of week-to-week variance were missing.
    - Bench players never filled in for a starter who sat.
  - All three are fixed. Calibration slopes are now 0.62 / 1.02 / 0.99, and ACCEPT trades now predict 0.82–0.91 wins against 0.86–1.05 real, which is slightly conservative.
- **What still doesn't.**
  - Weekly head-to-head between balanced teams is close to a coin flip for any model; the best Brier is about 0.23.
  - The calculator ranks trades about as well as comparing total rest-of-season value. Its added value is honest magnitudes, not better ordering.
  - Before the season, ESPN's own projections rank teams slightly better than the simulator.
- **Bugs found.**
  - A crash in the live app. With the new availability calibration on, `build_state` reused the name `team_played` for an int inside the player loop. It never reached users, but would have on the first refresh.
  - Two backtest bugs that misreported accuracy:
    - The preseason world dropped players who were injured on opening night.
    - The free-agent pool peeked at who plays later in the season.
  - A plotting bug in the trade-test summary.

## 2. How it was tested

- **Seasons.** Three test seasons: 2023-24, 2024-25 and 2025-26. For each test season:
  - The projection engine is refitted on the two seasons before it.
  - Season-model calibration constants were fitted on 2022-23. The matchup shrink β is fitted rolling-origin, on every earlier season.
  - Every prediction uses only box scores dated before it.
- **Leagues.** Mock 12-team leagues are snake-drafted from ESPN's preseason projections and scored with this league's settings. The truth comes from real box scores:
  - Each team starts its best available lineup each day.
  - Real head-to-head results decide the wins.
- **Trades.** Random 1-1, 2-1, 1-2 and 2-2 trades are made at weeks 6 and 12. Each one is replayed with the same forced drops and free-agent signings the calculator assumes, then compared with what the real box scores give.
- **Size.** 1,620 weekly matchups and 400 trades per season for the trade calculator; 280 matchups and about 22,000 player-games of availability for the simulator; 10 mock leagues for the league forecast.

## 3. Trying to break it

Thirteen adversarial tests (`audit/breaks.py`, run in CI by `tests/test_breaks.py`), all passing:

| Test | Result |
|---|---|
| A better player never lowers your expected wins | 1.000 → 1.215 |
| Swapping identical clones changes nothing | Δ = 0.0000 for both teams |
| Two-team league is zero-sum | +0.189 / −0.189 |
| Trading back restores the odds | +0.078 then −0.078 |
| A roster of injured players can't win | 0 points, 0% |
| Lineup rules enforced (13 centers) | 28 starts in 7 days, the maximum allowed |
| A week with no games | 50%, no crash |
| Zero variance | no division by zero |
| Doubling all scoring weights | probabilities unchanged |
| A superteam | > 99.9% |
| One 80-point game from a 20-point player | projection 20 → 28.6 (passes the < 30 test, see §7) |
| 2-for-1 trades create no phantom value | equal to the same roster change without trading |
| No peeking at the future | box scores after the date set to 999: projections identical |

The break tests didn't find the real bugs. Reading the code and the backtests did (§1).

## 4. Scoreboard: original model vs final model (same tests, same mock leagues)

| Metric | Goal | 2023-24 original | 2023-24 final | 2024-25 original | 2024-25 final | 2025-26 original | 2025-26 final |
|---|---|---|---|---|---|---|---|
| Weekly win chances (trade calculator): Brier | lower | 0.238 | **0.229** | 0.244 | **0.228** | 0.239 | **0.233** |
| Weekly win chances: calibration slope | 1.00 | 0.39 | **0.62** | 0.41 | **1.02** | 0.49 | **0.99** |
| Trades: slope of real on predicted Δwins | 1.00 | 0.71 | **0.99** | 0.45 | **0.72** | 0.62 | **0.97** |
| Trades: correlation | higher | 0.48 | 0.51 | 0.41 | 0.41 | 0.50 | 0.49 |
| ACCEPT trades: predicted gain (wins) | = real | 1.19 | 0.91 | 1.29 | 0.82 | 1.40 | 0.87 |
| ACCEPT trades: real gain (wins) | higher | 0.97 | 1.04 | 0.81 | 0.86 | 1.07 | 1.05 |
| DECLINE trades: real gain (wins) | | −0.67 | −0.82 | −0.32 | −0.32 | −0.65 | −0.63 |
| Rest-of-season wins after week 6: error | lower | 2.14 | **1.95** | 2.26 | **2.06** | 2.41 | 2.51 |
| Plays tonight? (availability): Brier | lower | 0.147 | **0.134** | 0.164 | **0.151** | 0.165 | **0.151** |
| Simulator: points per game, error | lower | 9.52 | 9.52 | 9.70 | 9.70 | 9.63 | 9.63 |
| Simulator: player-weeks inside 80% range | 80% | 77% | 77% | 77% | 77% | 80% | 80% |
| Simulator: weekly matchups, Brier | lower | 0.175 | 0.183 | 0.191 | 0.183 | 0.204 | 0.198 |
| Simulator: weekly matchups, calibration slope | 1.00 | 0.94 | 1.00 | 0.90 | 1.17 | 0.82 | 0.95 |
| League forecast preseason: rank correlation | higher | 0.42 | **0.51** | 0.08 | **0.19** | 0.11 | 0.06 |
| (benchmark) ESPN projections' ranking | | 0.55 | | 0.27 | | 0.01 | |
| League forecast after week 6: rank correlation | higher | 0.71 | 0.71 | 0.70 | 0.70 | 0.66 | 0.64 |
| After week 6: teams inside the 80% wins range | 80% | 83% | 82% | 88% | 84% | 80% | 78% |

Both columns use the fixed backtest harness, so the differences come from the model alone. With
the old harness, the preseason league forecast looked much worse: rank correlation 0.29 on 2023-24.

## 5. What works

1. **Player projections.** Per-game error is 9.5–9.7 fantasy points, against 9.9–10.1 for a "season average
   blended with last season" baseline, in all three seasons. Every component helps when tested on
   its own. For example, removing injury and teammate-absence information raises the bias from about 0 to −1.9.
2. **Simulator ranges.** 78–79% of player-games and 77–80% of player-weeks land inside the 80% range.
3. **Simulator matchups.** Brier 0.18–0.20, calibration slope 0.95–1.17.
4. **Rest-of-season FPPG blend.** (k·prior + n·live)/(k+n) beats the prior alone, last season alone and
   this season alone at every sample size. The error is flat for k between 6 and 9, so k stays at 6.
5. **Accounting.** Every property and break test passes. Trade deltas are exactly zero-sum in a two-team
   league, and a trade followed by its reverse cancels.
6. **The verdict is useful.** ACCEPT trades really gained 0.86–1.05 wins on average, and DECLINE trades lost
   0.32–0.82.

## 6. What didn't work, and what was changed

Each change below was fitted on a season before the test seasons, then judged on the test seasons.

### 6.1 Availability calibration (`model/availability.py: calibrated_rate`)

- **Problem.** The raw "games-played rate" was used as a probability. Players with low past rates played far more than
  predicted: 36% predicted vs 70% actual in the bottom bin. Players with high rates played less: 97% vs 94%.
  The model's Brier equalled the naive "everyone at the league rate" benchmark (0.147 vs 0.147).
- **Evidence.** On 2022-23, actual in-season availability ≈ 0.524 + 0.378 × rate: only 38% of a player's
  deviation from average carries forward. Preseason, last season's rate barely predicts this season's
  (slope 0.05) for fantasy-relevant players.
- **Fix.** A linear calibration, preseason and in-season, blended by team games played.
- **Effect.** Brier fell 8–9% in every season (0.147 → 0.134, 0.164 → 0.151, 0.165 → 0.151). Every
  reliability bin is now within about 5 points.

### 6.2 Missing weekly variance (`trade/matchup.py`)

- **Problem.** The weekly normal approximation treated each game as independent. On 2022-23, the real team-week
  residual s.d. was ≈ 185 points against 118 modelled.
- **Fix.** Two terms were added:
  - τ²U: a player's true level is uncertain, and the error is shared by all his games that week. It shrinks as he plays games.
  - ρR: absences cluster, since one injury costs several games.
- **Fitted values.** τ = 0.3, ρ = 2.0, on 2022-23.

### 6.3 Bench fill-ins (`trade/lineup.py: _fill_in_week`)

- **Problem.** The lineup started players on expected value. A started player who sat left his slot empty, so the
  model valued depth at zero. ESPN locks each player at his own tip-off: a manager benches a player who
  has been ruled out and starts the next-best one. The backtest truth does this too.
- **Fix.** An exact Poisson-binomial recursion, vectorized over the days of the week. It gives each
  player's P(starts and plays) when he only starts if fewer than S higher-ranked players are available.
  The same recursion carries the first two moments of the points, so the weekly variance includes the negative
  correlation between a starter sitting and his fill-in playing.
- **Verification.** The recursion matches brute-force enumeration exactly (unit test), and the normal
  approximation matches Monte Carlo within 2 points of probability.
- **Effect on 2022-23.** Trade correlation 0.42 → 0.45; the slope of real on predicted weekly points change 0.70 → 0.77.
- **Cost.** The trade finder takes 3.6 s per partner instead of 2.4 s.

### 6.4 Matchup shrink β, fitted rolling-origin

- **Problem.** After 6.1–6.3, β = 1 was calibrated on 2022-23 (slope 0.98), but the held-out seasons came
  out overconfident: slopes 0.56–0.75. 2022-23 is a poor season to fit on alone. Its "last season" is
  2021-22, when COVID protocols distorted games played.
- **Fix.** β is the probit slope of results on the model's z-score. Since M and V don't depend on β, it is
  refitted before each test season on every earlier season: 0.91, then 0.75, then 0.75. Live: 0.745,
  fitted on 2022-23 through 2025-26.
- **Effect.** Calibration slopes 0.62 / 1.02 / 0.99. 2023-24 stays somewhat overconfident because only
  2022-23 was available to fit on. Trade slopes 0.99 / 0.72 / 0.97, and ACCEPT predictions are now close to reality and slightly conservative.

### 6.5 Backtest harness fixes (these change what the tests report, not the live app)

- **Preseason world dropped injured players.** The simulator's preseason world held only players who
  appeared in their team's first 5 games. A star hurt on opening night therefore scored 0 for the whole season.
  Fixed: every fantasy-relevant player ESPN projected is in the world, and availability decides how much he plays.
  Preseason rank correlation on 2023-24 went from 0.29 to 0.51, against 0.55 for ESPN's ranking.
  The live simulator builds rosters from ESPN and was never affected.
- **Free-agent look-ahead.** The season-model harness's free-agent pool only held players who played later
  that season, which flattered free agents. Removed.
- **ESPN games blend.** The harness now applies ESPN's projected-games blend exactly as the live model does.
- **Season form re-tuned.** With the preseason world fixed, the league forecast's season-long form spread
  was re-tuned on 2024-25 (80% ranges covering 80%): 0.30 → 0.20. On 2025-26 the 80% wins ranges
  cover 74–85% across checkpoints, and the champion Brier beats a coin flip at every checkpoint
  (Forecast tests tab).
- **Plot bug.** The trade scatter used `DataFrame.shape` instead of the `shape` column.

### 6.6 Live-app bug

In `build_state`, `team_played` was a dict of team games played and was then reassigned to an int
inside the player loop. With the availability calibration on, the second player would have crashed
the app. Renamed.

## 7. Tried and rejected

| Idea | Result | Decision |
|---|---|---|
| Pull simulator FPPG toward ESPN's projection ("anchor"), weight 0 / 0.5 / 0.8 / 1.0 | preseason rank corr 0.29 / 0.26 / 0.20 / 0.20 (2023-24) | off (weight 0) |
| Heavier ESPN projected-games weight preseason, 0.5 / 0.75 / 1.0 | 0.291 / 0.298 / 0.319 (before the world fix); 0.505 vs 0.515 after | kept 0.5 (gain within noise) |
| Absence-streak availability: a player who has missed N straight games plays less, by streak × horizon | a strong pattern (21+ straight missed: plays only 10–33% of the expected games) but the season-model Brier got worse (0.2309 → 0.2331 to 0.2360) | not used; the live app has ESPN's injury report for this |
| Add players hurt at the prediction date back into in-season simulator worlds, with streak-based availability | week-6 rank corr 0.715 → 0.678, week 12 0.794 → 0.788 | not adopted |
| Winner's-curse discount on the free agent a 2-for-1 signs | real/predicted for the best free agents: 0.93, 0.99, 1.14 by season: no stable effect | not adopted |
| 2-for-1 trades are overrated (2022-23: predicted +0.2, real −0.56) | didn't replicate: −0.15, +1.04, +0.41 in the test seasons | sampling noise |
| Season before last in the rates prior | slightly worse projections | not used |

## 8. What still doesn't work, and suggested next steps

1. **Weekly matchups are inherently close to coin flips.** Brier is about 0.23 against 0.25 for always saying 50%.
   The model's probabilities are now honest, but they'll never be extreme.
2. **Trade ranking.** The correlation of predicted with real Δwins (0.41–0.51) equals that of simply comparing
   total rest-of-season value (0.48–0.51). Most of the outcome is noise: injuries and hot streaks after
   the trade. Treat the verdict as "is the price fair", and the size of the number as an honest estimate.
3. **Preseason rankings.** The simulator (0.51 / 0.19 / 0.06) is close to ESPN's projection ranking
   (0.55 / 0.27 / 0.01) but not clearly better. Early in the season, read the league forecast together
   with ESPN's projections. By week 6 the forecast is clearly better (0.64–0.71 vs 0.54–0.67).
4. **Simulator team totals run 2–4% low (−22 to −40 points a week).** The linear availability calibration
   slightly under-predicts the most durable players (top bin 90–91% predicted vs 93–95% actual). *Suggested:*
   an isotonic or logit-scale calibration, split by minutes.
5. **One huge game moves a projection a lot.** One 80-point game takes a 20-point player to 28.6. *Suggested:*
   winsorize single games at about 2.5 s.d. in the live average.
6. **Injury reports aren't backtestable yet.** No historical injury reports exist. The app now records the
   daily report (`status_snapshots`). After this season, the injury model (§3 of the methodology) can be
   tested the same way.
7. **Lineup management.** Fill-ins assume every absence is known before lock (ψ = 1). For a manager who
   doesn't set lineups daily, depth is worth less; set `lineup.fill_in` below 1.

## 9. Configuration changes (`tradecalc/config/league.yaml`)

| Setting | Before | After |
|---|---|---|
| `model.availability_calibration` | none (raw rates) | pre 0.735 + 0.047·h; in-season 0.524 + 0.378·rate |
| `model.proj_uncertainty` (τ) | none | 0.3 |
| `model.absence_cluster` (ρ) | none | 2.0 |
| `model.matchup_shrink` (β) | 1 (none) | 0.745 |
| `lineup.fill_in` (ψ) | none (no fill-ins) | 1.0 |
| `model.espn_anchor.weight` | none | 0.0 (tested, rejected) |
| `SEASON_LAYER.talent_sd` (sim/engine.py, league forecast season form) | 0.30 | 0.20 (re-tuned on 2024-25 after the preseason-world fix) |
