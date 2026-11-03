// Answers Discord interactions from the snapshot the daily GitHub job writes (tradecalc/discord_bot.py):
// /odds team opponent [week], /standings (so far, with expected wins), /forecast (projected final standings),
// /matchups [week], and team-name autocomplete. The odds are the app's own numbers, computed each morning
// for every pair of teams and every week left.

export const PING = 1;
export const COMMAND = 2;
export const AUTOCOMPLETE = 4;
const PONG = 1;
const MESSAGE = 4;
const CHOICES = 8;
const EPHEMERAL = 64;
const COLOR = 0xf26522;

// Same rounding as Python's format() (half to even), so the bot and the weekly post agree.
function round(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

export function pct(p) {
  if (p == null) return "–";
  if (p < 0.005) return "<1%";
  if (p > 0.995) return ">99%";
  return `${round(p * 100)}%`;
}

const num = (x) => round(x).toLocaleString("en-US");
const md = (s) => String(s).replace(/([\\*_~`|>])/g, "\\$1");
const norm = (s) => String(s).toLowerCase().replace(/[^a-z0-9]/g, "");

function reply(data) {
  return { type: MESSAGE, data: { allowed_mentions: { parse: [] }, ...data } };
}

function privately(content) {
  return reply({ content, flags: EPHEMERAL });
}

function option(interaction, name) {
  return (interaction.data.options || []).find((o) => o.name === name)?.value;
}

// Teams matching what someone picked or typed: an autocomplete pick is the team id; otherwise the
// closest name match (exact, then starts-with, then contains), all of them if it's ambiguous.
export function findTeams(snap, value) {
  if (value == null) return [];
  const v = String(value).trim();
  const byId = snap.teams.filter((t) => String(t.id) === v);
  if (byId.length) return byId;
  const q = norm(v);
  if (!q) return [];
  for (const test of [(n) => n === q, (n) => n.startsWith(q), (n) => n.includes(q)]) {
    const hits = snap.teams.filter((t) => test(norm(t.name)));
    if (hits.length) return hits;
  }
  return [];
}

function weekOf(snap, number) {
  return snap.weeks.find((w) => w.number === number);
}

function pickTeam(snap, value, label) {
  const hits = findTeams(snap, value);
  if (hits.length === 1) return { team: hits[0] };
  const names = (hits.length ? hits : snap.teams).map((t) => t.name).join(", ");
  return { error: hits.length ? `"${value}" matches more than one team: ${names}.` : `I couldn't find ${label} "${value}". Teams: ${names}.` };
}

function noWeek(snap, week) {
  const w = weekOf(snap, week);
  if (w && w.status === "final") return `${w.label} is over. \`/matchups week:${week}\` shows its results.`;
  const open = Object.keys(snap.odds).map(Number).sort((a, b) => a - b);
  return open.length ? `There's no week ${week} left. Odds are available for weeks ${open[0]}–${open[open.length - 1]}.`
    : "The season is over.";
}

function odds(snap, interaction) {
  const a = pickTeam(snap, option(interaction, "team"), "team");
  if (a.error) return privately(a.error);
  const b = pickTeam(snap, option(interaction, "opponent"), "team");
  if (b.error) return privately(b.error);
  if (a.team.id === b.team.id) return privately("Pick two different teams.");
  const week = option(interaction, "week") ?? snap.current_week;
  const wk = String(week);
  if (!snap.odds[wk]) return privately(noWeek(snap, week));
  const ia = snap.team_ids.indexOf(a.team.id);
  const ib = snap.team_ids.indexOf(b.team.id);
  const p = snap.odds[wk][ia][ib];
  const pa = snap.proj[wk][String(a.team.id)];
  const pb = snap.proj[wk][String(b.team.id)];
  const A = md(a.team.name);
  const B = md(b.team.name);
  const label = weekOf(snap, week)?.label ?? `week ${week}`;
  const real = (snap.schedule[wk] || []).some(([h, w]) => (h === a.team.id && w === b.team.id) || (h === b.team.id && w === a.team.id));
  let text = `🏀 **${A}** has a **${pct(p)}** chance of beating **${B}** in ${label}.`;
  text += `\nProjected: ${A} ${num(pa.mean)} · ${B} ${num(pb.mean)}`;
  if (pa.banked || pb.banked) text += ` (so far ${num(pa.banked)} – ${num(pb.banked)})`;
  text += real ? (week === snap.current_week ? "\nThat's their matchup this week." : `\nThat's their week ${week} matchup.`)
    : "\nThey don't play each other that week: this is a what-if.";
  text += `\n-# Odds as of ${snap.as_of_label}`;
  return reply({ content: text });
}

function table(snap, title, description) {
  return reply({ embeds: [{ title: `${snap.league}: ${title}`, color: COLOR, description,
    footer: { text: `As of ${snap.as_of_label}` } }] });
}

function matchups(snap, interaction) {
  const week = option(interaction, "week") ?? snap.current_week;
  const wk = String(week);
  const label = weekOf(snap, week)?.label ?? `Week ${week}`;
  const games = snap.text.matchups[wk];
  const done = snap.text.results[wk];
  if (!games && !done) return privately(`No matchups for week ${week}.`);
  return reply({ embeds: [{ title: `${label}: ${games ? "matchups" : "results"}`, color: COLOR, description: games || done,
    footer: { text: `As of ${snap.as_of_label}` } }] });
}

function autocomplete(snap, interaction) {
  const focused = (interaction.data.options || []).find((o) => o.focused);
  const q = norm(focused?.value ?? "");
  const choices = snap.teams.filter((t) => !q || norm(t.name).includes(q)).slice(0, 25)
    .map((t) => ({ name: String(t.name).slice(0, 100), value: String(t.id) }));
  return { type: CHOICES, data: { choices } };
}

export async function handleInteraction(interaction, getSnapshot) {
  if (interaction.type === PING) return { type: PONG };
  let snap;
  try {
    snap = await getSnapshot();
  } catch {
    if (interaction.type === AUTOCOMPLETE) return { type: CHOICES, data: { choices: [] } };
    return privately("I couldn't load the projections just now. Try again in a minute.");
  }
  if (interaction.type === AUTOCOMPLETE) return autocomplete(snap, interaction);
  if (interaction.type === COMMAND) {
    const name = interaction.data.name;
    if (name === "odds") return odds(snap, interaction);
    if (name === "standings") return table(snap, "standings", snap.text.standings);
    if (name === "forecast") return table(snap, "projected final standings", snap.text.forecast);
    if (name === "matchups") return matchups(snap, interaction);
  }
  return privately("I don't know that command.");
}
