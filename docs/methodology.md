# Methodology

This document specifies every model in the calculator: estimators, equations, parameters and
how each was fitted and validated. Code references are relative to `tradecalc/`. Current
parameter values live in `config/league.yaml` (season model, calibration) and
`config/sim_params.yaml` (projection engine, refit with `tradecalc sim-fit`).

---

## 0. Architecture

Two models share data, the injury model and the league settings:

| | **Season model** (`model/`, `trade/`) | **Projection engine** (`sim/`) |
|---|---|---|
| Unit | player: fantasy points per game $\mu_i$, spread $\sigma_i$, availability $a_i(d)$ | player: per-minute stat rates, minutes, availability; simulated box scores |
| Week | analytic: Gaussian team total | Monte Carlo over every NBA game |
| Used by | Matchup tab, Trade builder, Trade finder, Players, quick league forecast | Simulate tab, detailed league forecast; its minutes feed the season model |
| Speed | ~10 ms per trade (enumerates ~10⁵ trades) | ~1 s per 10,000 weeks |

Fantasy points are linear in the box score: $\mathrm{FP} = w^\top x$ with league weights
$w$ (PTS +1, REB +1, AST +2, STL +4, BLK +4, TO −2, FGM +2, FGA −1, FTM +1, FTA −1, 3PM +1).

---

## 1. Season model

### 1.1 Prior (preseason view of a player)

Three estimates of fantasy points per game are averaged (missing ones dropped):

$$\text{prior}_i = \operatorname{mean}\{\, m_i\, c(r_i),\;\; \hat f^{\text{ESPN}}_i,\;\; m_i\, \hat f^{\text{min}}_i \,\}$$

* **ADP curve** $c(r)$: monotone decreasing map from draft rank to FPPG, fitted on last season
  (rank → actual FPPG): 9-rank rolling mean, then isotonic regression weighted by games played
  (`model/prior.py`).
* **ESPN projection** $\hat f^{\text{ESPN}}$: ESPN's season projection scored with league weights.
* **Minutes view** $\hat f^{\text{min}} = \mathbb E[\text{FP}\mid \hat m_i]$: projected minutes × the
  engine's per-minute rates (§2.1–2.2), so role (starter / bench) moves the prior.
* $m_i$: research-based production multiplier in the first / second year back from a major
  injury (§3); not applied to ESPN's projection, which already reflects it.

### 1.2 Shrinkage of live data

$$\mu_i = \frac{k\,\text{prior}_i + n_i\, \bar x_i}{k + n_i}, \qquad k = 6$$

with $n_i$ games played this season and $\bar x_i$ their mean FP. **Role-change override**: if
$|\text{MPG}_{\text{last 5}} - \text{MPG}_{\text{season}}| / \text{MPG}_{\text{season}} > 0.20$
and $n_i \ge 5$, use $k/2$ and replace $\bar x_i$ by $\tfrac12(\bar x_i + \bar x_i^{\text{last 10}})$.

$k$ is chosen by backtest (§5.1): rest-of-season MAE is flat for $k \in [6, 9]$ in all three test seasons.

**Outlier games are not capped.** Winsorizing single games (at $\text{ref}\,(1 \pm z\,\text{cv})$ around the
prior or the running estimate, $z \in \{1.5, 2, 2.5, 3\}$) raised rest-of-season MAE by 2–9% in every
season 2022-23 to 2025-26: big games are mostly real signal (a bench player promoted to starter
routinely beats twice his preseason projection). Capping only games more than 25–45 points above
the running estimate changed MAE by at most ±0.1%.

### 1.3 Per-game spread

$$\sigma_i = \frac{k\, \text{cv}\, \mu_i + n_i\, s_i}{k + n_i}, \qquad \text{cv} = 0.42$$

$s_i$ = sample s.d. of the player's FP this season; cv = median coefficient of variation among the
top-156 players by FPPG (CV falls steeply with FPPG, so it is estimated on rosterable players).

### 1.4 Availability $a_i(d)$: probability of playing a game on date $d$

**Base rate** (calibrated by the audit, `model/availability.py: calibrated_rate`):

$$a^{(0)}_i = \alpha_0 + \beta_0\, h_i, \qquad
a^{(1)}_i = \alpha_1 + \beta_1\, \frac{10\, h_i + n_i}{10 + G_t}, \qquad
a_i = (1 - \omega)\, a^{(0)}_i + \omega\, a^{(1)}_i,\;\; \omega = \min(G_t/10, 1)$$

$h_i$ = games-played rate over the last two seasons with each season's longest absence removed;
$G_t$ = team games played this season. Fitted on 2021-22 → 2022-23:
$(\alpha_0, \beta_0) = (0.735, 0.047)$ (preseason: last season's rate barely predicts this season's
for fantasy-relevant players), $(\alpha_1, \beta_1) = (0.524, 0.378)$ (in-season: ~38% of a deviation
from average persists). Capped at 0.95.

*Tested and not adopted:* a logit form with expected minutes $\hat m_i$,
$\operatorname{logit} a^{(1)}_i = 1.158 + 0.362\operatorname{logit}(\text{rate}) + 0.827(\hat m_i/30 - 1)$
(`in_logit` in the config). It lowered the held-out availability Brier score in all three test seasons
(0.134 → 0.133, 0.151 → 0.150, 0.151 → 0.149) and fixed the under-prediction for the most durable
players, but trade decisions got slightly worse (correlation of predicted with real Δwins down in all
three seasons; real gain of the top third of predicted trades 1.36 / 1.17 / 1.33 → 1.36 / 1.12 / 1.13),
and the simulator's low team totals were only partly fixed.

**Injuries** (§3) modify $a_i(d)$: out ⇒ 0 until the return window; the probability of being
back ramps linearly from 0 at ESPN's return date $R$ to $\text{rtp}$ at $R + w$,
$w = \operatorname{clip}(0.3 \cdot \text{days out}, 4, 30)$; day-to-day ⇒ 0.5 until $R$; profile
multipliers apply in date windows (ramp-up weeks, year-1 / year-2 games effects).

**Second opinion on games**: expected remaining games
$E_i = \sum_j a_i(d_j)$ are blended with ESPN's projected games played $E^{\text{ESPN}}_i$ (scaled to
games left): $E'_i = (1-\lambda)E_i + \lambda E^{\text{ESPN}}_i$, $\lambda = 0.5\max(0, 1 - G_t/41)$,
implemented by rescaling the base rate (bounded to ×[0.7, 1.3]).

**Rest on back-to-backs** (engine only, §2.5).

### 1.5 Daily lineups with fill-ins

Each day $d$, with $A_d$ the players whose NBA team plays, the number of usable slots $S_d$ is the
size of the best legal lineup if all of $A_d$ played: the assignment problem

$$\max_{x}\; \sum_{i \in A_d}\sum_{s} x_{is}\, \mu_i \quad \text{s.t.}\; \sum_s x_{is} \le 1,\; \sum_i x_{is} \le 1,\; x_{is} = 0 \text{ if } i \text{ ineligible for slot } s$$

(PG, SG, SF, PF, C, G, F, 3×UTIL; Hungarian algorithm, `scipy.optimize.linear_sum_assignment`).

**Fill-ins (added by the audit).** ESPN locks each player at his own tip-off, so a manager who learns
a starter is out benches him and starts the next-best player. With players ranked by $\mu$ and
$\psi$ = share of absences known before lock (`lineup.fill_in`, 1.0), player $i$ *looks available*
with probability $r_i = 1 - \psi(1 - a_i)$ and starts if fewer than $S_d$ higher-ranked players look
available. With $K_{<i}$ that count (a Poisson-binomial variable),

$$P(i \text{ starts and plays on } d) = a_i(d)\; P(K_{<i} < S_d),$$

computed exactly by a recursion over $k \in \{0, \dots, S_d\}$ ($k = S_d$ absorbing), vectorized over
the days of the week (`trade/lineup.py: _fill_in_week`). Position rules enter through $S_d$ only
(the fill-in order ignores them; ESPN rosters with G/F/3×UTIL slots are flexible). The original
model started players on expected value $a_i\mu_i$ and left the slot empty when a starter sat, which
undervalued depth. Started games per player: $g_i = \sum_d r_i P(K_{<i} < S_d)$.

### 1.6 Weekly team distribution

With $q_{id} = a_i(d) P(K_{<i} < S_d)$, the weekly mean is $M = \sum_i \mu_i \sum_d q_{id}$. The game-level
variance is

$$V_{\text{games}} = \sum_i \sigma_i^2 \sum_d q_{id} \;+\; \sum_d \operatorname{Var}\!\Big(\sum_i \mu_i Z_{id}\Big),$$

where $Z_{id}$ indicates "starts and plays". The second term is exact: the same recursion carries the
first two moments of the accumulated points for each $k$, so the negative correlation between a
starter sitting and his fill-in playing is included (verified against brute-force enumeration in
`tests/test_lineup.py`, and the normal approximation matches Monte Carlo within 2 points of
probability on random matchups).

The audit found $V_{\text{games}}$ too small. Two terms were added (`trade/matchup.py`):

$$U = \sum_i \Big(\mu_i\, s_i \sum_d q_{id}\Big)^2, \quad s_i = \sqrt{\tfrac{k}{k + n_i}} \qquad \text{(uncertain true level, shared by a player's games)}$$

$$R = \sum_i v_i^2\, G_i (G_i - 1)\, \bar a_i (1 - \bar a_i) \qquad \text{(absences cluster within a week)}$$

$G_i = \sum_d P(K_{<i} < S_d)$ is the number of starts he would get if healthy, $\bar a_i$ his average
availability over them, and $v_i = (\mu_i - \mu_{(11)})^+$ his value over the best player outside
the top 10 (a sitting starter is replaced, so only the gap is lost). Then

$$V = V_{\text{games}} + \tau^2 U + \rho R .$$

### 1.7 Matchup and season

$$P(\text{win}_w) = \Phi\!\left(\beta\, \frac{M_{\text{you}} - M_{\text{opp}}}{\sqrt{V_{\text{you}} + V_{\text{opp}}}}\right),
\qquad \mathbb E[\text{wins}] = \sum_{w \ge \text{now}} P(\text{win}_w)$$

Points already scored in the current week (ESPN, days before today) are added to $M$.
$\tau = 0.3$ and $\rho = 2.0$ were fitted on 2022-23 mock leagues (§5) by weekly log-loss and
calibration slope. $\beta$ is the probit slope (no intercept) of real results on
$z = (M_{\text{you}} - M_{\text{opp}})/\sqrt{V_{\text{you}} + V_{\text{opp}}}$; because $M$ and $V$ do not
depend on $\beta$ it is refitted **rolling-origin** in the audit: before each test season, on every
earlier season from 2022-23. Fits: 0.91 (2022-23 only), 0.75 (through 2023-24), 0.75 (through
2024-25); live value $\beta = 0.745$ (2022-23 to 2025-26). 2022-23 alone is a poor fitting season:
its "last season" is 2021-22, when COVID protocols distorted games played.

### 1.8 Trade evaluation

For trade $T$, rosters are swapped, then each team's active roster (non-IR) is restored to 13:
excess ⇒ drop lowest ROS value $\mu_i \sum_j a_i(d_j)$; shortfall ⇒ sign the best free agent by
ROS value; incoming OUT players fill an empty IR slot. Then

$$\Delta = \mathbb E[\text{wins}\mid \text{after}] - \mathbb E[\text{wins}\mid \text{before}]$$

for both teams, recomputing every affected week (including weeks against the partner).
Verdict: ACCEPT if $\Delta_{\text{you}} \ge 0.10$, DECLINE if $\le -0.10$. Explanation reasons are
ranked by their value in season fantasy points (production gap × games, games gap × FPPG,
lineup crunch × weeks, games lost to injury × FPPG, ROS value of forced drops).

### 1.9 Trade finder

Enumerate all 1-for-1, 2-for-1, 1-for-2, 2-for-2 trades per partner (~9,500); prune by ROS value
(you may lose ≤ 5% of value sent; partner ≤ 15%); fully evaluate the best 300; keep
$\Delta_{\text{you}} \ge 0.10$ and $\Delta_{\text{partner}} \ge -0.10$; rank by $\Delta_{\text{you}}$.

---

## 2. Projection engine

### 2.1 Per-stat rates (empirical Bayes)

For each stat $s$ (per-minute FGA, FTA, REB, AST, STL, BLK, TOV; 3PA share of FGA; 2P%, 3P%, FT%)
with numerator $x$ and exposure $e$ (minutes or attempts):

$$\pi_i = \frac{K_s\, \bar r_{g(i)} + x^{\text{prev}}_i}{K_s + e^{\text{prev}}_i}, \qquad
\hat r_i = \frac{(K_s + 0.5\, e^{\text{prev}}_i)\,\pi_i + x^{\text{cur}}_i}{K_s + 0.5\, e^{\text{prev}}_i + e^{\text{cur}}_i}$$

$\bar r_{g}$ = position-group mean (G/F/C). $K_s$ is chosen by **split-half prediction** (odd games
predict even games, exposure-weighted squared error over a grid) on each fit season and averaged:
e.g. FGA 25 min, REB 38 min, STL 200 min, 3P% 300 attempts. (Adding the season before last to the prior
was tested and made projections slightly worse, so it is not used.)

### 2.2 Minutes

Expected minutes $\hat m_i$: EWMA ($\alpha = 0.25$) of this season's minutes, blended with a prior
$\hat m^{0}_i$ with weight $n/(n+3)$. Preseason prior = mean of available views: last season's
last-20-game MPG (same team; if last season was < 55% of the games of the season before — e.g. an
injury — that healthier season's MPG instead), the typical minutes of his ESPN depth-chart slot,
ESPN projected minutes. Current depth chart then bounds it: starters ≥ 26, depth 2 ≤ 30,
depth 3 ≤ 22, deeper ≤ 12. Each team's top 13 is trimmed bench-first to the 281 minutes real
rotations carry (fitted).

Per simulation, minutes follow a regression fitted on two seasons of as-of context
($R^2 \approx 0.1$; single-game minutes are noisy):

$$m = \hat m + c + \beta_{\text{same}} A_{\text{same}} + \beta_{\text{other}} A_{\text{other}} + \beta_{\text{b2b}} \mathbb 1_{\text{b2b}}
+ \beta_{\star}\, (|\text{margin}| - 12)^+ \mathbb 1_{\hat m \ge 30} + \beta_{\text{bench}}(|\text{margin}| - 12)^+ \mathbb 1_{\hat m < 20} + \varepsilon$$

$A$ = expected minutes of absent teammates (same position group / other); $\varepsilon \sim N(0, \sigma(\hat m))$
by minutes bucket (4.3–5.8). Truncated to [0, 48]; then each team's simulated minutes are rescaled to
$241 \times 0.933$ (share of team minutes played by the modelled rotation, fitted).

### 2.3 Usage redistribution

$$r_{is} = \hat r_{is}\,(1 + \gamma_s\, \text{miss}_{s}), \qquad \text{miss}_s = \frac{\sum_{j \text{ absent}} \hat r_{js}\hat m_j}{\sum_{j \in \text{rotation}} \hat r_{js}\hat m_j}$$

$\gamma_s$ by WLS of (game rate / baseline − 1) on miss share: FGA 0.37, FTA 0.55, AST 0.33, REB 0.11.

### 2.4 Opponent and pace

Team defensive factor per stat: $f_{ts} = \frac{K + n\, o_{ts}}{K + n}$ where $o_{ts}$ is the opponent
per-possession rate allowed relative to league; $K$ by split-half (3P% allowed: $K = 320$ games, i.e.
essentially noise; steals/turnovers forced: $K \approx 5$–$8$). Prior: last season's factor.
Game pace = own + opponent − league; pace ratio multiplies counting stats. Absent opposing
defenders: efficiency × $(1 + \sum_j \delta_j \hat m_j / 48 / \text{DRtg}_{\text{lg}})$, $\delta_j$ = last season's
on/off DRtg difference shrunk by $\text{MIN}/(\text{MIN} + 2000)$.

### 2.5 Availability in the engine

$P(\text{play}) = a_i(d) \times$ rest factor: logistic model of a healthy rotation player sitting one game,

$$\operatorname{logit} P(\text{sit}) = \theta_0 + \theta_1 \text{b2b} + \theta_2 (\text{age}-28)^+ + \theta_3\, \text{b2b}(\text{age}-28)^+ + \theta_4 \tfrac{\hat m}{30} + \theta_5\, \text{b2b}\tfrac{\hat m}{30}$$

plus a shrunk coach effect on back-to-backs; the factor is $P(\text{no sit}\mid \text{game}) / P(\text{no sit}\mid \text{average game})$.

### 2.6 Game simulation (`sim/engine.py: simulate_games`)

Per NBA game and draw: margin $\sim N(\mu_m, 14.05)$, $\mu_m$ from season net ratings × pace + 1.5 home;
pace shock $\sim N(1, 0.039)$; joint absences of both rotations; minutes (§2.2); then

$$\text{FGA} \sim \text{Pois}(\lambda), \;\; \text{3PA} \sim \text{Bin}(\text{FGA}, p_3^a),\;\; \text{2PM} \sim \text{Bin}(\text{FGA}-\text{3PA}, p_2),\;\; \text{3PM} \sim \text{Bin}(\text{3PA}, p_3)$$
$$\text{FTA} \sim \text{NB}(\lambda_{FT}, r = 3.4),\;\; \text{FTM} \sim \text{Bin}(\text{FTA}, p_{FT}),\;\; \text{REB, AST, STL, BLK} \sim \text{NB},\;\; \text{TOV} \sim \text{Pois}$$
$$\text{PTS} = 2\,\text{2PM} + 3\,\text{3PM} + \text{FTM}$$

$\lambda = \hat r \cdot m \cdot (\text{usage}) \cdot f \cdot \text{pace} \cdot \kappa \cdot \phi_i \cdot \zeta_i$:
$\kappa = 0.965$ global rate calibration (fitted on the prior season with true availability);
$\phi_i \sim N(1, 0.18)$ a weekly form factor shared by a player's games in the week (sized so weekly
player ranges cover 80% on the fit season); $\zeta_i$ the ESPN anchor (§2.8). NB sizes $r$ by method
of moments given the expected count.

### 2.7 Fantasy week

Per draw and day, the manager starts the best eligible lineup **among players active that day**
(ranked by expected FP; one assignment per distinct availability pattern), sums starters' FP and
compares teams. Points already scored this week are added.

### 2.8 ESPN anchor (tested by the audit, switched off)

The anchor scales counting stats so a player's FPPG moves toward ESPN's preseason projection,
fading with games played:

$$\zeta_i = \operatorname{clip}\!\left(\frac{(1-w)\,\text{FP}^{\text{eng}}_i + w\,\text{FP}^{\text{ESPN}}_i}{\text{FP}^{\text{eng}}_i}, 0.6, 1.6\right), \qquad w = w_0\,\frac{10}{10 + n_i}$$

On the tuning season (2023-24) every $w_0 > 0$ made the preseason team ranking worse
(Spearman 0.29 at $w_0 = 0$, 0.26 at 0.5, 0.20 at 0.8 and 1.0), so $w_0 = 0$
(`model.espn_anchor.weight`). The real cause of the weak preseason ranking was a backtest bug (§5).

### 2.9 Season layer (league forecasts)

Per simulated season: form $\exp(z\, s_i - s_i^2/2)$, $z \sim N(0,1)$, $s_i = 0.20\sqrt{20/(20 + n_i)}$;
and a two-state weekly injury chain with long-absence share $\pi = 0.175$ and mean spell 3.5 weeks
(measured on box scores): $P(\text{stay out}) = 1 - 1/3.5$,
$P(\text{start}) = \pi(1 - P_{\text{stay}})/(1 - \pi)$; per-game availability outside spells is scaled by
$1/(1-\pi)$ so mean availability is unchanged. Form size tuned on the prior season so 80% win ranges
cover 80%.

---

## 3. Injury model

ESPN injury report (status, return date, type, news) + history (injury log kept by the app, ESPN
season outlooks, long absences in last season's box scores, manual CSV). Text is matched to ~20
profiles (`config/injury_profiles.yaml`) built from published NBA studies, each giving typical days
out, $P(\text{return this season})$, and year-1 / year-2 multipliers on production, games and spread,
plus a ramp-up window. Details and sources: `docs/injury_research.md`. A player's normal
games-played rate excludes each season's longest absence so injuries are not double counted.

### 3.1 Historical injury reports (collected for backtesting)

`ingest/injury_reports.py` (`tradecalc injury-reports`) downloads the NBA's official pre-game injury
report (the 5 PM ET PDF, about two hours before most tip-offs) for every game day of 2021-22 to
2025-26: 816 reports, about 92,000 rows (game, team, player, status, reason), stored in
`data/seed/injury_reports_<season>.csv.gz`. Checks: every report parsed, every team mapped, 90–96% of
names match a box-score player (the rest mostly missed the whole season), and players listed Out
played 0.1% of the time. How often players really played, by their 5 PM status (G League
assignments excluded):

| Status | 2021-22 | 2022-23 | 2023-24 | 2024-25 | 2025-26 | all | rows |
|---|---|---|---|---|---|---|---|
| Out | 0.3% | 0.2% | 0.1% | 0.0% | 0.1% | 0.1% | 41,658 |
| Doubtful | 5.0% | 4.7% | 1.5% | 2.6% | 1.8% | 3.0% | 1,524 |
| Questionable | 52% | 51% | 56% | 50% | 55% | 52% | 11,561 |
| Probable | 92% | 87% | 90% | 91% | 92% | 90% | 4,335 |
| Available | 81% | 84% | 82% | 89% | 88% | 85% | 4,191 |

The live model gets ESPN's feed, which only says "day-to-day", and uses 50% for all of them: right for
Questionable, far too high for Doubtful and too low for Probable.

---

## 4. League forecast

Repeat $N$ times (10,000 quick / 2,000 detailed): finished weeks = real results; each remaining
week draws every team's score (season model: $N(M, V)$ × a persistent team form; engine: §2.6–2.9),
decides scheduled matchups (tie = ½), ranks by wins then points. Outputs: mean wins, 10th–90th
percentile, $P(\text{1st})$, $P(\text{top 3})$, $P(\text{last})$, mean rank.

---

## 5. Validation protocol (`tradecalc audit`, `audit/`)

**Out of sample by construction.** For test season $S \in \{2023\text{-}24, 2024\text{-}25, 2025\text{-}26\}$,
engine parameters are fitted on $S-2, S-1$; the availability calibration, $\tau$ and $\rho$ were fitted on
2022-23 only, which is before every test season, and $\beta$ on every season before $S$ (§1.7);
every prediction uses only data dated before it. A leakage test overwrites all post-date box scores
with 999 and checks projections are unchanged. Two settings of the league forecast (ESPN anchor,
ESPN games weight) were chosen on 2023-24, so for those two only 2024-25 and 2025-26 are fully held out.

**Mock leagues.** 12 teams × 13 players, snake draft by ESPN's preseason projection (pick among best 3,
p = .6/.25/.15), league scoring and slots, 24-week round robin. Ground truth from real box scores:
each day the best available lineup among players who actually played (greedy by position group,
ranked by FPPG through the prior day). As-of states (`audit/hist_state.py`) are built with the real
trade-calculator code; the free-agent pool holds every player known at that date (ESPN projection
or games played), without peeking at who plays later in the season.

**Harness bugs found and fixed by the audit.** (1) The engine's preseason world held only players
who appeared in their team's first 5 games, so a star injured on opening night scored 0 all season;
fixing it raised the preseason rank correlation on 2023-24 from 0.29 to 0.51 (ESPN's ranking: 0.55).
(2) The season-model harness's free-agent pool only contained players who played later that season
(a look-ahead that flattered free agents); removed. (3) The harness now applies ESPN's projected-games
blend exactly as the live model does.

| Test | Metric |
|---|---|
| 5.1 Season model, rest-of-season FPPG | MAE after N = 5/10/20/30 games: prior, live, blend over $k$ |
| 5.2 Engine point projections | MAE / RMSE / bias per player-game vs FPPG baseline (prior season shrunk, $k=6$) |
| 5.3 Engine ranges | share of actual FP inside simulated 50% / 80% intervals (game and player-week) |
| 5.4 Availability | Brier of P(play) for rotation players; calibration table |
| 5.5 Weekly matchups (engine, random rosters; season model, mock leagues) | Brier, log-loss, logistic calibration slope (1 = honest, < 1 = overconfident), reliability table |
| 5.6 League forecast | Spearman of predicted vs final order, MAE wins, 80%-range coverage, champion Brier vs 1/12 |
| 5.7 **Trade counterfactuals** | random trades at weeks 6 and 12; realized Δwins = real wins of the post-trade rosters (same drops / FA signings) minus real wins of the originals; OLS slope of realized on predicted (1 = calibrated), correlation, mean realized for ACCEPT |
| 5.8 Properties | monotonicity, zero-sum, reversibility, lineup legality, degenerate inputs, scale invariance, no leakage |

Metric definitions: Brier $= \frac1n\sum (p - y)^2$ (coin flip on balanced matchups ≈ 0.25);
log-loss $= -\frac1n \sum y\log p + (1-y)\log(1-p)$; calibration slope = coefficient on
$\operatorname{logit} p$ in a logistic regression of $y$; coverage = share inside the stated interval;
trade slope = OLS coefficient of realized on predicted Δwins (regression dilution from noisy
predictions pushes it below 1 even for an unbiased model, so ≈ 0.8–1.2 is the realistic target).

## 6. Known limitations

* Weekly head-to-head between balanced teams is close to a coin flip (best achievable Brier on
  balanced mock leagues ≈ 0.23); probabilities are honest, but no model makes them extreme.
* Trade **ranking** skill (correlation of predicted with realized Δwins ≈ 0.4–0.5) is about the same
  as comparing total rest-of-season value; the model's added value is calibrated magnitudes and the
  lineup / schedule / roster-spot accounting, not better ranking.
* Mock leagues have no trades, waivers or streaming; mock drafts only take players who played at
  least one game that season; lineups use a greedy position rule.
* No historical injury reports exist, so injury-report effects are validated only through research
  (§3), not backtested. In-season backtest worlds leave out players who are hurt at the prediction
  date (adding them back with a fitted absence-streak availability was tested: neutral to slightly
  worse, not adopted). Live, ESPN's injury report handles this.
* Minutes are the dominant error source (knowing true minutes cuts per-game MAE ~20%).
* Fill-ins assume every absence is known before lock ($\psi = 1$); a manager who doesn't set lineups
  daily gets less from depth (`lineup.fill_in` < 1).
