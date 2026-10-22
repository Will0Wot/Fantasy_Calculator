# Model audit report

*Scope: the season model (Matchup tab, Trade builder, Trade finder, quick league forecast), the
projection engine (Simulate tab, detailed league forecast), the availability/injury layer, and the
backtest harness itself. Every formula is in `docs/methodology.md`; the raw results are in
`tradecalc/config/audit.json` (final model), `audit_baseline.json` (original model) and `audit_box.json`
(final code with the old per-minute engine), and are re-created with `tradecalc audit --baseline`.*

## 0. October 2026 update: RotoBot-style possession simulator

*The whole audit was re-run twice on the same code: with the new engine (`config/audit.json`) and with the old
per-minute engine (`config/audit_box.json`). The Model audit tab shows both.*

**What changed**
1. **Games are simulated possession by possession** (methodology §2A), the way RotoBot describes its game
   simulations: who's on the floor, who shoots from where, who's guarding him, makes and misses, passes and
   rebounds. It's built from ESPN play-by-play (2021-22 on) and NBA tracking data.
2. **Every remaining game feeds the trade calculator** (§2A.5). Each morning every NBA game left in the season is
   simulated. Each player's projection for each game is then the average of his season-long number and that game's
   simulation. Matchups, trades, streaming and rest-of-season values all use it, and each night's bench order
   follows that night's projections.
3. **Availability, measured on five seasons of official injury reports.**
   - A player not on the report plays his next game 92% of the time (his season rate says 79%), fading back over
     about six games (§1.4).
   - In the simulator, absences persist within a week (§2.5).
4. **Simulator weekly form** was refitted for the new engine (sd 0.30, §2A.3), so its weekly ranges hold.
5. **Players the simulator barely knows** (§2A.2, §2A.5):
   - Players who missed most of last season get their own older seasons.
   - Rookies phase in over about two games.

**Results on the same code: old per-minute engine vs possession engine**

| Metric | Goal | 2023-24 per-minute | 2023-24 possession | 2024-25 per-minute | 2024-25 possession | 2025-26 per-minute | 2025-26 possession |
|---|---|---|---|---|---|---|---|
| Weekly win chances (trade calculator): Brier | lower | 0.229 | 0.224 | 0.228 | 0.224 | 0.233 | 0.230 |
| Weekly win chances: calibration slope | 1.00 | 0.62 | 0.85 | 1.02 | 1.15 | 0.99 | 1.07 |
| Trades: slope of real on predicted Δwins | 1.00 | 0.99 | 1.17 | 0.72 | 0.77 | 0.97 | 0.99 |
| Trades: correlation | higher | 0.51 | 0.53 | 0.41 | 0.45 | 0.49 | 0.50 |
| ACCEPT trades: predicted gain (wins) | = real | 0.91 | 0.80 | 0.82 | 0.84 | 0.87 | 0.93 |
| ACCEPT trades: real gain (wins) | higher | 1.04 | 0.96 | 0.86 | 0.84 | 1.05 | 1.00 |
| DECLINE trades: real gain (wins) |  | −0.82 | −0.85 | −0.32 | −0.45 | −0.63 | −0.61 |
| Preseason rest-of-season wins: error | lower | 3.06 | 2.83 | 3.30 | 3.06 | 3.20 | 3.04 |
| Rest-of-season wins after week 6: error | lower | 1.95 | 1.99 | 2.06 | 1.93 | 2.51 | 2.42 |
| Plays tonight? (availability): Brier | lower | 0.134 | 0.134 | 0.151 | 0.151 | 0.151 | 0.151 |
| Simulator, same player-games: per-minute engine error | lower | 10.13 | 10.13 | 10.20 | 10.20 | 10.04 | 10.04 |
| Simulator, same player-games: possession engine error | lower | 10.09 | 10.09 | 10.18 | 10.18 | 10.02 | 10.02 |
| Player-weeks inside the simulator's 80% range | 80% | 77% | 80% | 77% | 79% | 80% | 82% |
| Simulator: weekly matchups, Brier | lower | 0.192 | 0.181 | 0.192 | 0.187 | 0.189 | 0.220 |
| Simulator: weekly matchups, calibration slope | 1.00 | 1.11 | 1.19 | 1.20 | 1.25 | 1.32 | 0.79 |
| League forecast preseason: rank correlation | higher | 0.52 | 0.75 | 0.20 | 0.54 | 0.06 | 0.57 |
| (benchmark) ESPN projections' ranking |  | 0.55 | 0.55 | 0.27 | 0.27 | 0.01 | 0.01 |
| League forecast after week 6: rank correlation | higher | 0.71 | 0.69 | 0.70 | 0.74 | 0.64 | 0.65 |
| After week 6: teams inside the 80% wins range | 80% | 82% | 81% | 82% | 83% | 77% | 74% |

- **Better in all three seasons:**
  - The trade calculator's weekly win chances (Brier).
  - Trade ranking (correlation 0.51/0.41/0.49 → 0.53/0.45/0.50).
  - Preseason league forecasts. They now beat ESPN's own projection ranking every season: 0.75 / 0.54 / 0.57,
    against 0.55 / 0.27 / 0.01.
  - Preseason season-wins error.
  - The simulator's player-game error on identical games (0.02–0.04 points better).
  - Weekly ranges closer to 80%.
- **Mixed:**
  - Trades rated ACCEPT really gained 0.02–0.08 wins less in each season. That's within noise (about 200 such
    trades per season), while overall trade correlation went up.
  - The simulator's weekly matchups: Brier better in 2023-24 and 2024-25, worse in 2025-26 (0.189 → 0.220,
    slope 0.79).
  - Rest-of-season wins after week 6: better in two seasons out of three.
- **Matchup shrink:** with per-game projections, the live matchup shrink β refits to **0.778** (0.745 without).

**Measured directly**
- **Per-game projections** (next-week player-games, average miss):
  - The 50/50 average beat both the season model and the simulation alone, in all four seasons (§2A.5).
  - Season model / simulation / average: 9.81 / 9.86 / **9.71** (2022-23), 10.26 / 10.08 / **10.06**,
    10.25 / 10.18 / **10.11**, 10.13 / 10.04 / **9.97**.
- **Absences persist within a week.** Measured on 2022-23 and 2023-24:
  - Weekly games played vary 1.9× as much as independent games allow (2.8× with no injury report).
  - Fixing it, the actual/predicted spread of weekly margins went from 1.16–1.30 to 0.91–1.02.
  - The new engine's win-probability calibration on identical rosters went from 0.84 / 0.91 / 0.55 to
    1.26 / 1.01 / 0.98.
- **Not on the injury report:** 92.2% played against 78.7% predicted, with no back-to-back effect (92.2% vs
  92.3%). The miss-chance multipliers are 0.37, 0.62, 0.74, 0.82, 0.88, 0.94 for the next six games. Live
  only, so it's checked on the reports directly, not inside the audit.
- **Weekly form** (2022-23, known availability): sd 0.18 held 75% / 46% of player-weeks in the 80% / 50%
  ranges; 0.30 holds 79.5% / 50.3%.

**Bugs found and fixed**
- **Empty opening-week rotations.** Backtest worlds gave teams with fewer than 4 games an empty rotation (10
  teams in the first week of 2023-24).
- **Old engine fitted on the new one.** With the possession engine as the default, the per-minute engine's own
  constants (`rate_calibration`, form) were being fitted on possession simulations.
  - The committed parameters predate this, so the live app wasn't affected.
  - The next `tradecalc sim-fit` would have mis-scaled the season model's minutes view.
  - The first audit run had it too; its two affected tests were re-run on the fixed code.
- **File size.** The daily per-game projection file was 911 KB (committed every day); it's now 70 KB.

**Tried and rejected**
- **The season before last for everyone:** player-game error +0.02 in two test seasons out of three. Only used
  for players with almost no recent data.
- **Trusting thin play-by-play samples less:** the backtest favored the simulation even for thin samples. Only
  the rookie ramp was kept.
- **Weight 0.65 on the simulation instead of 0.5:** 0.01 points per game better. Kept 0.5.
- **A "stretch" correction because the simulation under-rates stars:** the 50/50 average isn't compressed
  (slope 0.94–0.97).

**Still open**
- **Bias when absences are known.** The new engine runs slightly high in two seasons: +0.5 and +0.9 points per
  game (+1.5 and +2.5 a week) in 2024-25 and 2025-26, −0.1 in 2023-24. With real, uncertain availability, its
  bias is −0.3 to −1.2.
- **2025-26 simulator matchups** were overconfident in the audit's test (slope 0.79), while the paired test on
  identical rosters gave 0.98. The audit's test redraws rosters each run, so ±0.01 of Brier is noise.

## 1. Summary

- **What works.**
  - Player point projections beat a season-average baseline in every test season.
  - The simulator's ranges are honest.
  - The simulator's weekly win probabilities are close to calibrated (slope 0.79–1.25 by season).
  - The trade calculator's accounting holds: rosters, drops, free agents, lineups and zero-sum. Every break test passes.
- **What didn't, and is fixed.**
  - The trade calculator's weekly win chances were badly overconfident: a calibration slope of 0.39–0.49, where 1.0 is honest.
  - Its trade gains were overstated: predicted gains for ACCEPT trades were 1.19–1.40 wins, against 0.81–1.07 real.
  - The causes:
    - Raw games-played rates were used as probabilities.
    - Two sources of week-to-week variance were missing.
    - Bench players never filled in for a starter who sat.
  - All three are fixed. Calibration slopes are now 0.85 / 1.15 / 1.07, and ACCEPT trades now predict 0.80–0.93 wins against 0.84–1.00 real.
- **What still doesn't.**
  - Weekly head-to-head between balanced teams is close to a coin flip for any model; the best Brier is about 0.23.
  - The calculator ranks trades about as well as comparing total rest-of-season value. Its added value is honest magnitudes, not better ordering.
  - Before the October 2026 update, ESPN's own projections ranked teams slightly better than the simulator
    before the season. The possession simulator now beats them in all three seasons (§0).
- **Bugs found.**
  - A crash in the live app. With the new availability calibration on, `build_state` reused the name `team_played` for an int inside the player loop. It never reached users, but would have on the first refresh.
  - Two backtest bugs that misreported accuracy:
    - The preseason world dropped players who were injured on opening night.
    - The free-agent pool peeked at who plays later in the season.
  - A plotting bug in the trade-test summary.
  - After the first push of this audit, a TypeError on the live site: Streamlit Cloud kept the old
    modules in memory (§6.7). Fixed so future pushes reload cleanly.

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
| Weekly win chances (trade calculator): Brier | lower | 0.238 | 0.224 | 0.244 | 0.224 | 0.239 | 0.230 |
| Weekly win chances: calibration slope | 1.00 | 0.39 | 0.85 | 0.41 | 1.15 | 0.49 | 1.07 |
| Trades: slope of real on predicted Δwins | 1.00 | 0.71 | 1.17 | 0.45 | 0.77 | 0.62 | 0.99 |
| Trades: correlation | higher | 0.48 | 0.53 | 0.41 | 0.45 | 0.50 | 0.50 |
| ACCEPT trades: predicted gain (wins) | = real | 1.19 | 0.80 | 1.29 | 0.84 | 1.40 | 0.93 |
| ACCEPT trades: real gain (wins) | higher | 0.97 | 0.96 | 0.81 | 0.84 | 1.06 | 1.00 |
| DECLINE trades: real gain (wins) |  | −0.66 | −0.85 | −0.32 | −0.45 | −0.65 | −0.61 |
| Preseason rest-of-season wins: error | lower | 3.18 | 2.83 | 3.35 | 3.06 | 3.18 | 3.04 |
| Rest-of-season wins after week 6: error | lower | 2.14 | 1.99 | 2.26 | 1.93 | 2.41 | 2.42 |
| Plays tonight? (availability): Brier | lower | 0.147 | 0.134 | 0.164 | 0.151 | 0.165 | 0.151 |
| Simulator, same player-games: per-minute engine error | lower | – | 10.13 | – | 10.20 | – | 10.04 |
| Simulator, same player-games: possession engine error | lower | – | 10.09 | – | 10.18 | – | 10.02 |
| Player-weeks inside the simulator's 80% range | 80% | 77% | 80% | 77% | 79% | 80% | 82% |
| Simulator: weekly matchups, Brier | lower | 0.175 | 0.181 | 0.191 | 0.187 | 0.204 | 0.220 |
| Simulator: weekly matchups, calibration slope | 1.00 | 0.94 | 1.19 | 0.90 | 1.25 | 0.82 | 0.79 |
| League forecast preseason: rank correlation | higher | 0.42 | 0.75 | 0.08 | 0.54 | 0.11 | 0.57 |
| (benchmark) ESPN projections' ranking |  | 0.55 | 0.55 | 0.27 | 0.27 | 0.01 | 0.01 |
| League forecast after week 6: rank correlation | higher | 0.71 | 0.69 | 0.70 | 0.74 | 0.66 | 0.65 |
| After week 6: teams inside the 80% wins range | 80% | 83% | 81% | 88% | 83% | 80% | 74% |

Both columns use the fixed backtest harness, so the differences come from the model alone. With
the old harness, the preseason league forecast looked much worse: rank correlation 0.29 on 2023-24.

## 5. What works

1. **Player projections.** Per-game error is 9.5–9.7 fantasy points, against 9.9–10.1 for a "season average
   blended with last season" baseline, in all three seasons. Every component helps when tested on
   its own. For example, removing injury and teammate-absence information raises the bias from about 0 to −1.9.
2. **Simulator ranges.** With the possession engine, 80–82% of player-games and 79–82% of player-weeks land
   inside the 80% range.
3. **Simulator matchups.** Brier 0.18–0.22, calibration slope 0.79–1.25 (§0).
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

### 6.7 Deploy bug: TypeError after a push

Streamlit Cloud pulls a new commit and re-runs `app.py`, but keeps the `tradecalc` modules it had
already imported. After this audit was pushed, the new `app.py` passed the matchup shrink to the old
four-argument `p_win` and the Matchup tab crashed with a TypeError. The app now fingerprints its source
and config files on every run; when they change, it drops the stale `tradecalc` modules and clears the
caches, so a push takes effect without a manual reboot. Verified headless: unchanged code keeps
the loaded modules, and a changed file reloads them.

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
| Cap single-game outliers at ref × (1 ± z·cv), z = 1.5–3, around the prior or the running estimate | rest-of-season FPPG error **2–9% worse** in every season 2022-23 to 2025-26: big games are mostly real (a bench player promoted to starter routinely beats twice his projection) | not adopted |
| Cap only games more than 25 / 35 / 45 points above the running estimate | error changed by at most ±0.1% | not adopted (no effect) |
| Availability on a logit scale with expected minutes (fitted on 2022-23) | availability Brier better in all 3 seasons (0.134 → 0.133, 0.151 → 0.150, 0.151 → 0.149); simulator low bias shrank (−32/−22/−40 → −31/−7/−34 pts a week) but persists; trade decisions slightly worse (correlation down in all 3 seasons; real gain of the top third of predicted trades 1.36/1.17/1.33 → 1.36/1.12/1.13) | not adopted (code kept, `in_logit` in the config) |

## 8. What still doesn't work, and suggested next steps

1. **Weekly matchups are inherently close to coin flips.** Brier is about 0.23 against 0.25 for always saying 50%.
   The model's probabilities are now honest, but they'll never be extreme.
2. **Trade ranking.** The correlation of predicted with real Δwins (0.41–0.51) equals that of simply comparing
   total rest-of-season value (0.48–0.51). Most of the outcome is noise: injuries and hot streaks after
   the trade. Treat the verdict as "is the price fair", and the size of the number as an honest estimate.
3. **Preseason rankings: fixed by the possession simulator.** The per-minute simulator (0.52 / 0.21 / 0.06)
   was close to ESPN's projection ranking (0.55 / 0.27 / 0.01) but not better. The possession simulator ranks
   teams at 0.75 / 0.54 / 0.57, better than ESPN in every season. By week 6 both are at 0.65–0.74.
4. **Simulator team totals ran 2–4% low with the per-minute engine** (−34 to −72 points a week on random
   fantasy rosters). The possession engine's are close: −27, +18 and +1 points a week in 2023-24, 2024-25 and
   2025-26 (it is calibrated to league stat totals). The per-minute engine's remain low; it is no longer the default.
5. **One huge game moves a projection a lot** (an 80-point game takes a 20-point player to 28.6). Capping
   was tested and makes projections worse (§7); the current behaviour is the better one on the data.
6. **Injury reports: now collected.** The NBA's official 5 PM report for every game day of 2021-22 to
   2025-26 (816 reports, ~92,000 rows) is in `data/seed/` (methodology §3.1). First finding: players listed
   Doubtful played 3% of the time, Questionable 52%, Probable 90%, while the model gives every
   "day-to-day" player 50% because ESPN's feed doesn't say which. *Next:* read the official report live
   for the finer status, and backtest injury durations and return ramps against these reports.
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
| `engine.mode` (October 2026) | box (per-minute) | possession (§0) |
| `engine.game_weight` | none | 0.5: each game = ½ season model + ½ that game's simulation |
| `engine.absence_rho` / `absence_rho_backtest` | none (independent games) | 0.59 live / 0.85 in backtests |
| possession `form_sd` (`sim/possession/engine.py`) | 0.18 | 0.30 |
| `model.matchup_shrink` (β), with per-game projections | 0.745 | 0.778 |
