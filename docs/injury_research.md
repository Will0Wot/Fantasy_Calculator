# NBA injury research behind the injury model

The calculator turns each injury into four questions:

1. **When is he back?** ESPN's expected return date, or the typical recovery time below.
2. **Will he come back this season at all?** Return-to-play (RTP) rate.
3. **How well does he play once back?** Production change in year 1 and year 2 after returning.
4. **How often does he play once back?** Games-played change, which reflects reinjury, rest and maintenance days.

The numbers live in `tradecalc/config/injury_profiles.yaml` and can be edited there.

## Summary table

"Year 1 / Year 2" columns are multipliers the model applies: 0.80 means 20% lower.

| Injury | Typical time out | RTP | Year-1 production | Year-1 games played | Year-2 production / games | Confidence |
|---|---|---|---|---|---|---|
| Achilles rupture | ~10–11 months | 75% | 0.80 | 0.80 | 0.90 / 0.90 | High |
| ACL tear | ~12 months | 85% | 0.83 | 0.78 | 0.95 / 0.92 | High |
| Patellar / quad tendon rupture | ~12 months | 75% | 0.90 | 0.75 | 0.95 / 0.85 | Medium |
| Knee cartilage / microfracture | ~5 months | 80% | 0.88 | 0.85 | 0.92 / 0.90 | Medium |
| Meniscus surgery | ~6 weeks | 91% | 0.82 | 0.80 | 0.92 / 0.85 | High |
| Lumbar disc / back surgery | 3–9 months | 86% | 0.90 | 0.85 | 0.95 / 0.92 | Medium |
| Foot fracture (Jones, navicular) | ~10 weeks | 88% | 1.00 | 0.90 | 1.00 / 0.95 | Medium |
| Shoulder surgery (labrum, cuff) | ~5 months | 94% | 0.95 | 0.92 | 1.00 / 0.97 | Low |
| Plantar fascia tear | ~5 weeks | ~98% | 0.97 | 0.93 | 1.00 / 1.00 | Low |
| Hand / wrist / finger fracture | ~5 weeks | 98% | 1.00 | 1.00 | 1.00 / 1.00 | High |
| Calf strain | 2–3 weeks | 100% | 0.98 | 0.94 | 1.00 / 0.98 | Medium |
| Hamstring strain | ~2 weeks | 100% | 0.96 | 0.92 | 1.00 / 0.97 | Medium |
| Groin / adductor / hip | ~2 weeks | 100% | 0.98 | 0.94 | 1.00 / 1.00 | Low |
| Concussion | ~1 week (3–5 games) | 100% | 1.00 | 0.99 | 1.00 / 1.00 | High |
| Ankle sprain | ~1 week (median 2 games) | 100% | 1.00 | 0.97 | 1.00 / 1.00 | High |
| Long absence last season, cause unknown | n/a | n/a | 0.95 | 0.92 | 1.00 / 1.00 | Low |

## What the research says, injury by injury

**Achilles tendon rupture.** This is the worst injury in basketball.
- Return rates in NBA cohorts range from 70% to 80%: 72% of 47 players, 80% of 25, 79.5% of 44, and 70.8% in a 348-player surgery study. Recovery averages about 311 days, or about 10.5 months.
- In the first season back, returning players started far fewer games (20.8 vs 49.4 the season before). PER dropped sharply right after return, and only 1 of 12 players in one study regained pre-injury form. Two seasons later, value over replacement was still about 24% lower than before.
- Players with a prior calf strain have higher odds of an Achilles rupture, and 86% of those ruptures were on the same side as the earlier strain.

**ACL tear.**
- 84–86% return, after about 372 days on average.
- In the first full season back, players played about 17 fewer games and had about 2.35 lower PER than matched controls, roughly 14% lower per-minute efficiency. Careers ended about 1.8 years earlier.
- Point guards and small forwards lost more minutes than centers did.

**Meniscus surgery.**
- 90.7% return. Players missed a median of 17 games.
- Year 1: games played fell from 79% to 61% of team games, minutes from 26.0 to 22.3, and PER from 16.4 to 14.6.
- Year 2: minutes and PER had mostly recovered, but players still appeared in fewer games (64%).
- About 21% needed another surgery on the same knee. Each extra year of age at surgery meant about 4.5% fewer games two years later.

**Patellar or quadriceps tendon rupture.**
- About 75% (patellar) to 78% (quadriceps) return, after about a year.
- Production returns to near pre-injury level within about two seasons. Games played stay lower for seasons 1 to 3.

**Knee cartilage / microfracture.** Among players who lasted two or more seasons after surgery, minutes fell by about 3 per game.

**Lumbar disc / back surgery.**
- About 86% of NBA players return (small sample), after 2.8 to 8.7 months.
- Performance afterward ranges from 64% to 104% of baseline.

**Foot fractures.**
- Jones fractures: return to play takes about 10 weeks, and 85–91% of basketball players get back to their previous level.
- Refracture is the main risk: 10% across 646 athlete fractures, and 3 of 10 in one NBA series.
- Production is largely unaffected once healed.
- No basketball data was found for navicular fractures. They are treated like Jones fractures.

**Hand / wrist fractures.** These have the highest return rate of any procedure studied (98.1%), with no meaningful effect on production.

**Hamstring strains.**
- A 2016–21 NBA study found a small but real drop in minutes and field-goal percentage after return.
- Reinjury rates were high: 81% of players had another injury (hamstring or elsewhere) that season or the next.

**Ankle sprains.** These are the most common NBA injury.
- 56% cost no games. The rest cost a median of 2 games.
- A previous ankle sprain raises the chance of another, and earlier foot, ankle, or knee injuries are linked to longer absences.

**Concussion.**
- Players miss about 3.5 games (7.7 days) under the current protocol.
- Matched-control studies found no measurable drop in performance after return.

**Age and prior injury.** A 17-year NBA review found no link between age and overall injury *rate*. Prior injury, especially to the same area, is the stronger predictor. Age matters more for recovery: older players are less likely to return and lose more games after meniscus surgery.

## How research numbers became model numbers

- **Production multiplier.** Fantasy points scale roughly with minutes × per-minute efficiency, so the year-1 multiplier is about (minutes ratio) × (PER ratio). For meniscus surgery: (22.3 / 26.0) × (14.6 / 16.4) ≈ 0.76, rounded to 0.82 because the cohort skews toward older role players.
  - It applies only to the **ADP-curve half** of a player's preseason estimate. ESPN's projection, the other half, already reflects its analysts' view of the injury.
  - Once real games are played, live stats take over within a few weeks, so the research matters most early in the season.
- **Games-played multiplier.** This is the relative drop in games played versus before the injury, for example 61% / 79% ≈ 0.78 for meniscus year 1.
  - To avoid counting an injury twice, the player's base games-played rate leaves out his longest absence each season. The profile then applies the research effect.
- **Return-to-play rate.** This caps the chance he is back at any point this season. The studies include fringe and older players who chose to retire, so the rates are conservative for stars.
- **Return dates slip.** Instead of switching a player on the day ESPN expects him back, the chance he is back rises from 0 on that date to the RTP rate over a window of 30% of his time out, between 4 and 30 days.
- **Ramp-up.** The first 1–4 weeks back count at 75–95% of normal output, reflecting minutes limits.
- **Volatility.** Year 1 after a major injury is 5–15% more volatile. This is an assumption: the studies report averages, not spread.

## Where injury history comes from

1. **ESPN injury report**, the live feed with return dates. Each refresh is stored, so a season's history builds up automatically.
2. **ESPN season outlooks.** The preseason notes flag, for example, Haliburton and Lillard (Achilles), Irving (ACL), and Kessler (shoulder surgery).
3. **Last season's game logs.** A run of 20 or more missed team games counts as a "long absence" when nothing more specific explains it.
4. **`tradecalc/data/injury_history.csv`** for anything you know that the sources above miss. Columns: `name,injury,returned`.
5. **`tradecalc/data/injury_overrides.csv`** to override a current injury. Columns: `name,status,games_out,return_date`.

## Limits

- The samples are small: Achilles studies cover 12 to 47 players, and some injury types have no NBA data. Those profiles are marked low confidence.
- The studies measure PER and minutes, not fantasy points, so the conversion is approximate.
- The ESPN injury feed covers only current injuries, so history before this season relies on outlooks and game logs.

## Sources

- [Glein et al., Achilles ruptures in NBA vs WNBA (PMC11452890)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11452890/)
- [Achilles rupture outcomes in NBA players, 1970–2018 (PMC7708715)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7708715)
- [Return to play, performance, and value of NBA players after Achilles rupture](https://lida.sport-iat.de/ta/Record/4076200)
- [Minhas et al., Effect of an orthopaedic surgical procedure in the NBA (AJSM 2016)](https://pubmed.ncbi.nlm.nih.gov/26801923/)
- [Prior calf strain and Achilles rupture risk in NBA players (Healio, 2026)](https://www.healio.com/news/orthopedics/20260924/nba-players-with-prior-calf-strain-had-increased-risk-for-achilles-tendon-rupture)
- [Achilles rupture in the NBA, 2000–2025 (OJSM 2026)](https://metatoc.com/papers/195974-achilles-tendon-rupture-in-the-nba-return-to-play-performance-decline-and-temporal-trends-2000-2025)
- [Kester et al., ACL reconstruction and career longevity in the NBA (KSSTA 2017)](https://www.springermedicine.com/athletic-performance-and-career-longevity-following-anterior-cru/20611376)
- [Performance after ACL reconstruction in NBA players (OJSM 2021)](https://pubmed.ncbi.nlm.nih.gov/33738307/)
- [Effect of ACL injury on post-recovery performance in the NBA (HSM 2024)](https://hsm.susu.ru/hsm/article/view/2579)
- [Nosrat et al., Meniscus surgery in NBA players (JSSM 2025)](https://www.jssm.org/jssm-24-363.xml-Fulltext)
- [Meniscal tears and NBA performance (Becker's)](https://www.beckersspine.com/uncategorized/nba-player-performance-not-usually-hurt-by-meniscal-tears/)
- [Patellar tendon repair in professional athletes](https://www.iat.uni-leipzig.de/datenbanken/iks/dbv-base-softball/Record/4063924)
- [NBA quadriceps tendon tears (ISAKOS 2023 poster)](https://www.isakos.com/Assets/Meetings/2023Congress/ePosters/ABS_23024_Tummala.pdf)
- [Microfracture outcomes in pro basketball](https://lermagazine.com/?p=2883)
- [Lumbar discectomy in professional athletes (PMC8873437)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8873437/)
- [Operative treatment of Jones fractures in the NBA](https://vivo.weill.cornell.edu/display/pubid26781131)
- [Jones fracture return to play meta-analysis (ACFAS)](https://www.acfas.org/return-to-play-and-fracture-union-after-the-surgical-management-of-jones-fractures-in-athletes-a-sy)
- [Most NBA players return from Jones fracture with no drop in performance](https://lermagazine.com/?p=25916)
- [Hamstring injury epidemiology in the NBA, 2016–2021](https://www.ejcrim.com/index.php/mltj/article/view/6764)
- [Ankle sprains in the NBA, 2013–2017](https://lida.sport-iat.de/dbb/Record/4056906)
- [NBA ankle injuries, 2015–2021 (PMC10387785)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10387785)
- [Injury in the NBA: a 17-year overview](https://vivo.weill.cornell.edu/display/pubid23015949)
- [Concussion return to play in the NBA, 1999–2018](https://vivo.weill.cornell.edu/display/pubid31276004)
- [Performance after a first concussion in NBA players](https://lida.sport-iat.de/ta/Record/4041667)
- [NBA meniscus study citing age and prior-injury effects (JSSM PDF)](https://jssm.org/24-2-363.p_d_f)
