"""Plain-language explanations shown in the app ("How this works" panels and the Help tab).

Written for someone who knows basketball but not statistics. Numbers come from the live
config, fitted parameters and saved validation results, so the text stays accurate.
"""
from __future__ import annotations


def _pct(x, d=0):
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


# --- per-tab panels -----------------------------------------------------------------------

def matchup(cfg) -> str:
    k = cfg["model"]["k"]
    return f"""
**What you're looking at:** this week's projected score for you and your opponent, your chance to win it,
and the same for every remaining week.

**How it's calculated (the quick "season model")**
1. **How good is each player?** Before the season it uses expectations: draft position, ESPN's projection, and
   his projected minutes (starter or bench) × his per-minute production.
   As games are played, his real numbers take over. After {k} games, real results count as much as the preseason
   view, and they quickly dominate after that.
2. **Will he play?** Each player gets a chance of suiting up, built from how often he's played the last two seasons,
   plus any current injury (see the Injuries tab).
3. **Who starts?** For each day, it picks your best legal lineup from players whose NBA team plays that day.
   Bench games don't count, so a crowded night can leave good players on the bench. If a starter is ruled out
   before tip-off, your next-best player with a game fills in, so depth is worth something.
4. **Win chance.** Your expected total and how much it could swing are compared with your opponent's.

**Dates:** only days from today onward are projected. Days already played this week use the real points ESPN
recorded.

**Injuries here:** a player who is out counts as 0 until his expected return. A day-to-day player is treated as a
coin flip for his next game. A player coming back from a major injury gets fewer expected games and a slightly
lower projection for his first year back.

For a more detailed game-by-game simulation of any matchup, use the **Simulate** tab.
"""


def league_forecast(cfg) -> str:
    return f"""
**What you're looking at:** the rest of the regular season played out thousands of times for all
{cfg['league']['num_teams']} teams. Each team's chance to win the league, finish top 3 or finish last, and its
projected number of wins.

**How it's calculated**
1. **Weeks already played** use the real results from ESPN. This week's points so far are counted.
2. **Every remaining week,** each team's score is simulated:
   - *Detailed:* every NBA game is played out (minutes, injuries, rest, teammates, opponents), and each team
     starts its best lineup each day.
   - *Quick:* each team's score is drawn from its projected weekly total and normal week-to-week swing.
3. **Every scheduled matchup is decided** (a tie counts as half a win), and wins are added up.
4. **There are no playoffs,** so the team with the most wins takes the league. Ties are broken by total points.
5. **Season-long luck:** in each simulated season a player can have a good or bad year, and can miss a multi-week
   stretch. Both were measured from real box scores, and the size of the "good or bad year" effect was tuned so
   the forecast's ranges proved honest on past seasons (see **Forecast tests**).
6. **Repeat** 2,000 (detailed) or 10,000 (quick) times and count how often each team finishes where.

**Reading it:** "Win league 35%" means that team finished first in 35% of simulated seasons. The heatmap shows
the full spread of where each team ends up. Early in the season the ranges are wide; they tighten as weeks are
played.

**The last-place line** goes to the team ranked last by average finish. It's a projection, not a sentence.
"""


def simulate(params: dict | None) -> str:
    form = (params or {}).get("form_sd", 0.18)
    return f"""
**What you're looking at:** any two fantasy teams, any week, played out 10,000 times, game by game.

**What happens in each simulated week**
1. **Who suits up.** For every NBA game, each rotation player's chance of playing comes from injuries, the injury
   report, and rest on back-to-backs. Older, heavy-minute players sit more often on the second night.
   When a star sits in a simulation, his teammates absorb his role in that simulation.
2. **How the game goes.** Each game gets a final margin (blowouts send starters to the bench early) and a pace
   (fast games mean more stats for everyone).
3. **Minutes.** Recent minutes, plus a share of any absent teammate's minutes (mostly to the same position), plus
   blowout effects. The team total always adds up to a real game's 240.
4. **Usage.** When teammates are out, the players left take more shots, free throws and assists.
5. **Opponent.** Some defenses force turnovers, some allow free throws, and some play fast. If the opponent's best
   defender sits, scoring gets slightly easier.
6. **The box score.** Every shot, make, rebound and assist is drawn, so a 25-point scorer sometimes drops 40 and
   sometimes goes cold. Each player also gets a "form" factor for the week (about ±{form:.0%}), so his good and
   bad nights cluster the way they do in real life.
7. **Your lineup.** Each day the best eligible players who are active are started. Then all the points are added
   up and compared.

**Current starting lineups.** Every team's starters come from ESPN's depth chart as it stands today. A listed
starter gets at least starter-level minutes, and backups are capped by their depth slot (6th man ~30, third
string ~22, deeper ~12). The rotation is then trimmed to a real game's total, cutting the deep bench first. Once
this season's games are played, a player's real minutes take over. The lineups used are shown under each
simulation.

**Players hurt last season but healthy now** aren't judged on their injury-shortened year:
- *Minutes* come from their last healthy season (plus the depth chart and ESPN), not from limited games.
- *A long absence with no known cause* doesn't lower their projected production.
- *Known major injuries* (an ACL or Achilles tear, for example) still carry the research-based effects on
  games played.
- *Per-minute production* still comes from last season. Testing showed that adding the season before made
  projections slightly worse, even for players coming off injury.

**Dates:** only the remaining days of the week are simulated. Points from days already played are added as they
actually happened. The schedule grid shows ● for games still to play and ○ for games already played.
"""


def trade_builder(cfg) -> str:
    t = cfg["trade"]
    return f"""
**How a trade is judged:** one question, *how many more weekly matchups will I win the rest of the season?*

1. **Swap the players** you picked.
2. **Fix both rosters.** Each team keeps a full active roster:
   - The team with too many players drops its least valuable ones.
   - The team with an open spot signs the best available free agent.
   - Injured players can move into an open IR slot instead of taking a roster spot.
3. **Replay the rest of the season** for both teams, before and after the trade: every remaining week, against
   the opponent actually on the schedule, with daily lineups. If you play your trade partner later, that week uses
   their new roster too.
4. **Compare expected wins.** The verdict uses the change in *your* expected wins:
   - **Accept:** you gain at least {t['verdict_accept']:.2f} wins.
   - **Decline:** you lose at least {abs(t['verdict_decline']):.2f} wins.
   - **Close call:** anything in between.

**Why the numbers can surprise you**
- **Games beat averages.** A 40-point player who plays 55 more games can be worth more than a 45-point player
  who is often hurt.
- **Lineup crunch.** If your lineup is already full most nights, an extra good player mostly sits on your bench,
  so the trade gains less than the raw numbers suggest.
- **Roster spots matter.** A 2-for-1 frees a spot for a free agent. A 1-for-2 forces you to drop someone.

**How injuries affect a trade**
- **An injured player you receive** counts as 0 until his expected return. After that he's eased back in rather
  than switched on at full strength, because return dates often slip.
- **There's a chance he doesn't come back this season.** For example, about 25% after an Achilles tear and about
  15% after an ACL tear, based on NBA studies.
- **Year one after a major injury,** he's expected to play fewer games and produce a bit less (see the Injuries tab
  for the research behind each injury type).
- **Day-to-day tags** barely matter over a season. They're shown, but ranked low.

**Reading the result**
- **Reasons** are ordered by how many fantasy points each is worth over the season: ▲ helps you, ▼ hurts you.
- **Weekly win chances** show each week before → after, e.g. 46% → 39% (−7%).
- **The partner's change** tells you whether they'd likely say yes. If they gain a lot, the deal probably leans
  their way.
"""


def trade_finder(cfg) -> str:
    t = cfg["trade"]
    return f"""
**Pick any team** in "Find trades for" (it starts on your team), then one partner or all of them. Every
partner appears in the results summary, including teams where no good trade exists.

**How trades are found**
1. **Try every small trade** with each team: 1-for-1, 2-for-1, 1-for-2 and 2-for-2. That's roughly 9,500
   combinations per team.
2. **Quick screen.** It drops trades that clearly lose you value, or that clearly cost the other team too much,
   using each player's rest-of-season value (points per game × expected games).
3. **Full evaluation** of the best {t['max_full_evals_per_opponent']} survivors per team, exactly as in the Trade
   builder (rosters fixed, rest of season replayed for both teams).
4. **Keep** trades that the Trade builder would call **Accept** (you gain at least
   {t.get('finder_min_gain', t['verdict_accept']):.2f} expected wins) and where the other team loses no more than
   {t['acceptance_threshold']:.2f} wins. That's a stand-in for "they might say yes."
5. **Rank** by how much you gain.

**Injuries** are handled exactly as in the Trade builder. Injured players are valued only for the games they're
expected to play, so the finder may suggest buying a hurt star cheaply, or selling someone whose injury risk it
rates higher than his owner might.

**Keep in mind:** "might accept" is the model's judgment, not the other manager's. A real person may value
names, upside or team needs differently. Use the list as a starting point for offers.
"""


def players(cfg) -> str:
    return f"""
**Columns**
- **FPPG:** projected fantasy points per game. Before the season it's the preseason view; it shifts toward his
  real numbers as games are played ({cfg['model']['k']} games = half real, half preseason).
- **Role / MIN:** projected minutes per game and role: **Starter** (first on ESPN's depth chart or 28+ minutes),
  **Rotation** (18–28) or **Bench** (under 18). Minutes come from this season's games, ESPN's depth chart, last
  season and ESPN's projected minutes, and are trimmed so each team's rotation adds up to a real game.
- **Prior:** the preseason view on its own, an average of three views: a draft-position curve, ESPN's projection,
  and projected minutes × the player's per-minute production (so a bench role pulls the projection down).
- **GP:** games played this season.
- **Avail.:** chance he plays his team's next game (injuries, injury report, normal games-missed rate).
- **Games left:** his NBA team's remaining regular-season games.
- **ROS value:** rest-of-season value, FPPG × expected games he'll actually play. Used to rank players and to
  screen trades quickly.
- **Notes:** injury status, injury history (e.g. "year 1 back from ACL tear"), and role changes (a big jump or
  drop in recent minutes).

**Replacement level** (bottom line) is the best free agent at each position. It's what an empty roster spot is
worth, because you could always pick that player up.
"""


def injuries(cfg) -> str:
    return """
**How injuries enter the model**

1. **Current injuries.** ESPN's injury report gives a status and usually an expected return date.
   - **Out:** 0% until the return date. Then his chance of playing rises over the next few days to weeks (longer
     for longer injuries), because return dates usually slip.
   - **Out for the season:** 0% for every remaining game.
   - **Day-to-day:** 50% for his next game (or until the listed return date), then normal.
2. **What kind of injury.** Each injury is matched to one of about 20 research-based profiles. Each profile sets:
   - typical time out;
   - the chance he returns this season at all;
   - how his production, games played and consistency change in **year 1 and year 2** after he returns.

   For example, after an Achilles tear: about 75% return, with about 20% lower production and 20% fewer games in
   year one. After an ankle sprain: almost no lasting effect.
3. **Easing back in.** For the first 1–4 weeks after returning, output is reduced, reflecting minutes limits.
4. **Injury history.** Players coming back from last season's major injuries are found through ESPN's
   preseason notes, long absences in last season's box scores, the app's own injury log (it builds up all season),
   and anything you add below.
5. **No double counting.** A player's normal games-played rate leaves out his one big absence each season, because
   the injury profile already accounts for it.

**Columns:** *Now / +30 / +90 days* show his chance to play his team's next game on or after that date, so you can
see a player's expected return.

**Corrections** override the model: a status or return date, or a past injury it doesn't know about. On the
shared league copy they're read-only (the commissioner maintains them). The research and sources are in
`docs/injury_research.md`.
"""


# --- Help & accuracy tab -------------------------------------------------------------------

def help_overview(cfg) -> str:
    return f"""
### What this tool does
It answers two questions for your league ({cfg['league']['num_teams']} teams, head-to-head points,
{cfg['league']['regular_season_weeks']} weeks, no playoffs):

1. **Will I win this week?** (Matchup and Simulate tabs)
2. **Does this trade help me win more weeks?** (Trade builder and Trade finder)

It doesn't just compare averages. It accounts for **how many games** each player plays, **whether he'll be
healthy**, **whether he'd even make your starting lineup** that night, and **who you play each week**.

### Two engines
| | Season model | Game-by-game simulator |
|---|---|---|
| Used in | Matchup, Trade builder, Trade finder, Players | Simulate |
| Speed | Instant, so it can test thousands of trades | About a second for 10,000 simulated weeks |
| Detail | Points per game, availability, daily lineups, schedule | Adds minutes, usage, opponent defense, pace, blowouts, rest, every stat |

Both share the same injury model and the same league settings.
"""


def help_trades(cfg) -> str:
    return trade_builder(cfg) + "\n" + trade_finder(cfg)


def help_accuracy(cfg, validation: dict | None) -> str:
    k = cfg["model"]["k"]
    parts = ["""
### Why you can trust it

*For the full data-science audit (three past seasons, before/after numbers, attempts to break the model) see the
**🔬 Model audit** tab; every formula is in **📐 Methodology**.*

The best test of a forecast is to give it only what was known at the time, and check it against what actually
happened. That's how this model was tested: on the entire **2025-26 season**, with every prediction made using
only games played *before* it, and settings fitted on the two seasons before that.
"""]
    if validation:
        pts, cov, mu = validation.get("points", {}), validation.get("coverage", {}), validation.get("matchups", {})
        base, eng = pts.get("baseline", {}), pts.get("engine", {})
        exact = pts.get("engine_rates_x_actual_min", {})
        parts.append(f"""
**1. It predicts players better than a season-average approach.** Over {validation.get('n_player_games', 0):,}
real player-games, the simulator's average miss was **{eng.get('MAE', 0):.2f} fantasy points per game**, against
{base.get('MAE', 0):.2f} for "season average blended with last season." Big misses shrank too (RMSE
{eng.get('RMSE', 0):.2f} vs {base.get('RMSE', 0):.2f}). Each piece (injuries, teammates, opponent) improved the
result when tested on its own. Most of the remaining miss is minutes: if minutes were known in advance, the miss
would drop to {exact.get('MAE', 0):.2f}. Single-game minutes are genuinely unpredictable, which is why the
simulator uses ranges rather than a single number.

**2. Its ranges are honest.** When the simulator says "80% chance he scores between X and Y," that should happen
80% of the time:
- single games: **{_pct(cov.get('game_in80'), 1)}** inside the 80% range, {_pct(cov.get('game_in50'), 1)} inside the 50% range;
- whole weeks per player: **{_pct(cov.get('week_in80'), 1)}** inside the 80% range, {_pct(cov.get('week_in50'), 1)} inside the 50% range;
- average bias: {cov.get('game_bias', 0):+.2f} points per game, essentially none.

**3. Its win probabilities mean what they say.** On {mu.get('n', 0)} test matchups between random fantasy rosters
in real 2025-26 weeks:
""")
        def _rng(s):
            lo, hi = (float(x) for x in s.strip("([]").split(","))
            return f"{lo * 100:.0f}–{hi * 100:.0f}%"
        rows = "\n".join(f"| {_rng(b['range'])} | {b['n']} | {_pct(b['predicted'])} | {_pct(b['actual'])} |"
                         for b in mu.get("buckets", []))
        parts.append("| Predicted win chance | Matchups | Average prediction | Actually won |\n|---|---|---|---|\n" + rows)
        parts.append(f"""
Brier score **{mu.get('brier', 0):.3f}**: lower is better, and a coin flip scores 0.250. Predictions near 50%
line up well. The further a prediction is from 50%, the more it tends to be somewhat overconfident: a "30%" week
was won more often than 30%. The likely reason is that these tests had no injury reports, so real weeks had more
surprise absences than the simulator expected. The live tool does use injury reports. Still, read a 25% or 75%
as "clearly behind / ahead, but very winnable / losable."

*Last measured {validation.get('run_on', 'n/a')}. Re-run with `tradecalc sim-validate`.*
""")
    else:
        parts.append("*Validation results haven't been saved yet. Run `tradecalc sim-validate` to measure them.*")
    parts.append(f"""
**4. The league forecast was tested on whole seasons.** Mock 12-team leagues from the real 2025-26 season were
forecast at preseason and after weeks 6, 12 and 18, and compared with how they actually finished. This
uncovered (and fixed) overconfidence: season-long luck (multi-week injuries, players having a good or bad year)
was missing. Now about 72–82% of teams finish inside the "80%" range. See the **🧪 Forecast tests** tab.

**5. The season model was tuned the same way.** How fast real results override preseason expectations ({k} games)
was chosen by testing several settings on last season, and picking the one that best predicted each player's
remaining-season scoring.

**6. Every behavior comes from data, not guesses:**
- **Back-to-backs:** healthy players sit about 9% of second nights vs 5% otherwise, and about 17% for players
  over 34.
- **Teammates out:** when 20% of a team's shots are missing, each remaining player shoots about 7% more.
- **Coaches:** rotation habits (how much starters play) carry over year to year and are used. Rest habits mostly
  don't, so they're heavily discounted.
- **Defense:** 3-point defense turned out to be mostly luck and is nearly ignored. Steals and turnovers forced are
  real.
- **Injuries:** effects come from published NBA studies (see `docs/injury_research.md`).

### What it can't do
- **Predict new injuries.** It knows the odds, not who will get hurt.
- **Read a coach's mind on minutes.** That's the biggest source of surprise in any single game.
- **Know which five share the floor.** It knows who's active, not the exact rotations within a game.
- **Separate probable / questionable / doubtful.** ESPN only reports "day-to-day" (treated as 50%). The app logs
  statuses daily so this can be measured during the season.
- **Read the other manager.** "Might accept" is the model's view of fairness, not theirs.
""")
    return "\n".join(parts)


def help_glossary() -> str:
    return """
### Glossary
- **FPPG:** fantasy points per game.
- **Expected wins:** the sum of your win chances across the remaining weeks. 60% + 70% + 40% = 1.7 expected wins.
- **ROS value:** rest-of-season value, FPPG × games he's expected to actually play.
- **Availability:** chance a player suits up for a given game.
- **Replacement level:** what the best free agent gives you, i.e. what an empty roster spot is worth.
- **Banked points:** points already scored this week. Only the remaining days are projected.
- **80% range:** in 8 of 10 simulated outcomes, the result lands inside it.
- **Back-to-back:** a team playing on consecutive nights. The second night carries more rest risk.

### Keeping it current
- **Refresh data** (sidebar) pulls new games, injuries, rosters and this week's scores. It also updates hourly on
  its own.
- `tradecalc sim-fit` refits the simulator on the latest seasons. Worth doing every few weeks.
- `tradecalc sim-validate` re-tests accuracy and updates the numbers on this page.
"""
