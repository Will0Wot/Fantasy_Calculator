# Production plan: any league, any platform

Status: phases 1–5 built and tested on branch `dev/multi-league` (November 2026); see §7. The live app keeps running from `main` until each phase
passes its tests and is merged.

## 1. Goal

Anyone can open the app, enter their fantasy basketball league from ESPN, Yahoo or Sleeper, pick their team, and
get every feature built so far for their own league:

- this week's matchup odds and the rest of the season;
- the league forecast;
- the trade builder and the trade finder;
- streaming;
- player values and injuries;
- the season tracker.

The recommendations have to make sense to an experienced manager. No "drop a star for a streamer".

## 2. What has to change

| Area | Today | Needed |
|---|---|---|
| League | One ESPN league, hard-wired in `config/league.yaml` (ID, scoring, roster slots, weeks) | Any league, with settings read from the platform |
| Platforms | ESPN only | ESPN (public and private), Sleeper and Yahoo, behind one adapter interface |
| Players | ESPN player IDs everywhere | ESPN's NBA player pool as the common ID; other platforms matched by name and team |
| Scoring | Projections stored in this league's fantasy points | Projections stored per stat, so any points scoring works |
| App | Loads the one league | League picker (platform + league ID or link), bookmarkable links, cached per league |
| Tracker, Discord bot | One league, files in the repo | Unchanged for the home league at first; a database later for other leagues |

### Model problems found (3 Nov 2026, week 3 streaming plans for all 12 teams)

1. **Injured stars look droppable.** The planner chose to drop Jimmy Butler (39.1 fantasy points a game last
   season), Brandon Ingram (35.6) and Jalen Suggs (33.8) for Neemias Queta (24.5). A player who's out gets a
   projection for when he returns of 21–27 a game, for two reasons:
   - the projection engine gives him 12 bench minutes, since he isn't in today's rotation;
   - the injury discount is applied on top of a draft position that already priced the injury in.
2. **Nobody claims a dropped injured star in the model.** It assumes a rival only claims a player who beats
   their weakest player now, but a rival with an open injured-reserve slot stashes him for free.
3. **A few bad games outweigh years of history.** The prior is worth 6 games for everyone, so after 7 bad games
   Mikal Bridges (33.3 last season, five straight 82-game seasons) is projected at 22.5. DeRozan (32.2),
   Wiggins (32.0) and Grayson Allen (30.5) show up as drop candidates too. The value 6 was fitted on all
   players together; a long, stable record should count for more than a rookie's.

## 3. Decisions (defaults; change any of them)

| Decision | Default | Why |
|---|---|---|
| Platforms | ESPN, Sleeper, Yahoo | The three biggest for fantasy basketball. Sleeper's API is public. Yahoo needs a Yahoo developer app (free) and each user signs in with Yahoo. |
| League formats | Head-to-head points (and season points totals) | This is what every feature models today. Category leagues need different math for every feature; they're listed as a later phase and the app says so clearly. |
| Accounts | None: a league is a link (`?platform=…&league=…&team=…`) | Nothing to store or secure. Private-league logins stay in the visitor's browser session, never on the server. |
| Hosting | Streamlit Community Cloud (free), plus a Dockerfile so it can move | The heavy NBA-wide work already runs once a day on GitHub Actions; each league only adds a few seconds. |
| Storage | None needed for the core features. Tracker and Discord bot for other leagues need a database (free Postgres) | They're the only features that must remember things between visits. |

## 4. Design

```
NBA layer (shared by every league, refreshed daily)          League layer (per league, cached ~1 hour)
  box scores (stats.nba.com, ESPN fallback)                    adapter: ESPN | Sleeper | Yahoo
  ESPN player pool: IDs, positions, injuries, projections        -> settings: scoring, roster slots, weeks, ties, moves
  NBA schedule, injury reports                                    -> teams, rosters, matchups, results
  possession projections per stat (daily job)                  league config = model settings + league settings
                                                               build_state(league config, NBA layer) -> every feature
```

- **Adapters** turn a platform's data into the same `LeagueData`:
  - settings: scoring, slots, eligibility, weeks with dates, ties, weekly move limit, injured-reserve slots;
  - teams and rosters;
  - matchups and results;
  - each player's positions on that platform and its injury status.

  Unknown scoring rules (e.g. double-double bonuses) are listed as warnings rather than silently ignored.
- **One player ID.** Every platform's players are matched to ESPN's NBA pool by normalized name and NBA team,
  with tests on the whole Sleeper player list. Anyone unmatched is reported.
- **League config:** `league.yaml` keeps the model settings. The league-specific sections (`league`,
  `scoring`, `roster`, `lineup`, `weeks`, `streaming.moves_per_week`) are built from the platform for each
  league, so most of the code doesn't change.

## 5. Phases

Each phase ends with every test passing and an end-to-end app check, then a merge to `main` (with your OK).

1. **Recommendations that make sense** (the problems above). **Done**: see `docs/model_audit.md` §10. It also
   found and fixed an ADP bug that undervalued most players outside the top 130 by 2–5 points a game.
   - Projections for injured players use their healthy minutes, and the injury discount is applied once.
     Checked against players who came back from long absences in 2022-23 to 2025-26.
   - How much a player's history counts depends on how long and steady it is. Fitted and tested out of sample
     on past seasons.
   - The rival-claim model includes injured-reserve stashes.
   - A guardrail: never suggest dropping a player whose record is clearly better than the best free agent's.
     Each drop candidate's track record is shown in the app.
   - Rerun the streaming plans for all teams, and the model audit, to confirm.
2. **Platform layer**
   - Neutral settings and `LeagueData`.
   - The ESPN adapter reads settings from ESPN (weeks from ESPN's own calendar), so any ESPN league works.
   - Per-league config and per-stat projections.
3. **Multi-league app**
   - League picker and private ESPN leagues.
   - A per-league cache and memory limits.
   - Old `?team=` links keep working.
   - CLI `--platform/--league`.
4. **Sleeper**: adapter, player matching and tests on recorded responses, plus a live test on a real league.
5. **Yahoo**: sign-in with Yahoo, adapter and tests. Needs a Yahoo developer app.
6. **Hardening and release**
   - Platform outages and bad input handled with clear messages.
   - Rate limits.
   - CI on every push.
   - A staging copy of the app.
   - A user guide and a privacy note.

Later:
- category leagues;
- the tracker and Discord bot for any league (needs the database);
- Fantrax.

## 6. Testing

- Unit tests for every adapter, built from recorded real responses (no network in tests).
- Property checks on recommendations:
  - no suggested drop of a player well above the free agents;
  - a trade's verdict flips sign when the trade is reversed;
  - win chances add up to 1.
- Historical backtests (`tradecalc audit`) before and after each model change, with the numbers in
  `docs/model_audit.md`.
- An app smoke test (Streamlit AppTest) for each platform, using fixture leagues.
- CI runs all of it on every push.

## 7. Status (3 Nov 2026)

| Phase | State | Tested how |
|---|---|---|
| 1. Sound recommendations | Done | Backtests on 2021-22 to 2025-26 (`docs/model_audit.md` §10); all 12 teams' streaming plans and the trade finder rerun on live data |
| 2. Platform layer | Done | ESPN's settings for the home league match `league.yaml` exactly; unit tests |
| 3. Multi-league app | Done | App smoke test on the demo league (first visit, team pick, every tab), cache keys per league, private-league isolation |
| 4. Sleeper | Built | Offline tests on recorded-shape responses; 478 of 554 active Sleeper players match ESPN's pool. **Needs a live league to confirm the scoring keys.** |
| 5. Yahoo | Built | Offline tests on documented-shape XML; sign-in flow tested in the app. **Needs a Yahoo developer app and a live league.** |
| 6. Hardening | Mostly | CI, clear errors, per-league caches. Memory: about 1.25 GB for the first league, about 20 MB for each more. |

**Not built yet:**

- category and roto leagues;
- the season tracker and Discord bot for leagues other than the home league (both need a database);
- Fantrax.
