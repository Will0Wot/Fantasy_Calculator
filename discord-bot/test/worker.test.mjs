// End-to-end tests of the Worker as Discord calls it: signed requests in, interaction responses out.
// Run with `npm test` (Node 20+, no packages needed).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import worker from "../src/worker.js";
import { COMMANDS } from "../src/commands.js";
import { pct } from "../src/bot.js";

const snapshot = JSON.parse(readFileSync(new URL("./snapshot.fixture.json", import.meta.url)));
const pair = await crypto.subtle.generateKey({ name: "Ed25519" }, true, ["sign", "verify"]);
const publicHex = Buffer.from(await crypto.subtle.exportKey("raw", pair.publicKey)).toString("hex");
let url = 0;
const env = () => ({ DISCORD_PUBLIC_KEY: publicHex, SNAPSHOT_URL: `https://data.test/${++url}.json` });

globalThis.fetch = async (u) =>
  String(u).includes("broken") ? new Response("nope", { status: 503 }) : Response.json(snapshot);

async function call(interaction, { badSignature = false, e = env() } = {}) {
  const body = JSON.stringify(interaction);
  const timestamp = String(Math.floor(Date.now() / 1000));
  const sig = await crypto.subtle.sign("Ed25519", pair.privateKey, new TextEncoder().encode(timestamp + body));
  const request = new Request("https://bot.test/", {
    method: "POST", body,
    headers: { "Content-Type": "application/json", "X-Signature-Timestamp": timestamp,
      "X-Signature-Ed25519": badSignature ? "ab".repeat(64) : Buffer.from(sig).toString("hex") },
  });
  const response = await worker.fetch(request, e);
  return { status: response.status, json: response.status === 200 ? await response.json() : null };
}

const command = (name, options = []) => ({ type: 2, data: { name, options } });
const opt = (name, value, extra = {}) => ({ name, value, ...extra });

test("answers Discord's PING and refuses bad signatures", async () => {
  assert.deepEqual((await call({ type: 1 })).json, { type: 1 });
  assert.equal((await call({ type: 1 }, { badSignature: true })).status, 401);
  const missing = await worker.fetch(new Request("https://bot.test/", { method: "POST", body: "{}" }), env());
  assert.equal(missing.status, 401);
});

test("/odds gives the snapshot's chance, both ways round", async () => {
  const { json } = await call(command("odds", [opt("team", "1"), opt("opponent", "2")]));
  const p = snapshot.odds["2"][0][1];
  assert.equal(json.type, 4);
  assert.match(json.data.content, new RegExp(`\\*\\*Inevitable\\*\\* has a \\*\\*${pct(p)}\\*\\* chance of beating \\*\\*Not\\\\_Eating\\\\\\*Poop\\*\\*`));
  assert.match(json.data.content, /That's their matchup this week\./);
  assert.deepEqual(json.data.allowed_mentions, { parse: [] });
  const back = await call(command("odds", [opt("team", "2"), opt("opponent", "1")]));
  assert.match(back.json.data.content, new RegExp(`has a \\*\\*${pct(1 - p)}\\*\\* chance`));
});

test("/odds understands typed names and what-if matchups in later weeks", async () => {
  const { json } = await call(command("odds", [opt("team", "inevit"), opt("opponent", "ballers club"), opt("week", 3)]));
  const p = pct(snapshot.odds["3"][0][3]);
  assert.ok(json.data.content.includes(`**Inevitable** has a **${p}** chance of beating **Ballers Club** in Week 3 (Nov 9–15).`));
  assert.match(json.data.content, /what-if/);
});

test("/odds explains mistakes privately", async () => {
  const same = (await call(command("odds", [opt("team", "1"), opt("opponent", "1")]))).json.data;
  assert.equal(same.flags, 64);
  assert.match(same.content, /two different teams/);
  const unknown = (await call(command("odds", [opt("team", "Lakers"), opt("opponent", "1")]))).json.data;
  assert.match(unknown.content, /couldn't find team "Lakers"\. Teams: Inevitable, /);
  const vague = (await call(command("odds", [opt("team", "ball"), opt("opponent", "1")]))).json.data;
  assert.match(vague.content, /matches more than one team: Ball Knowers, Ballers Club/);
  const late = (await call(command("odds", [opt("team", "1"), opt("opponent", "2"), opt("week", 9)]))).json.data;
  assert.match(late.content, /no week 9 left\. Odds are available for weeks 2–3/);
  const over = (await call(command("odds", [opt("team", "1"), opt("opponent", "2"), opt("week", 1)]))).json.data;
  assert.match(over.content, /Week 1 \(Oct 26–Nov 1\) is over\. `\/matchups week:1` shows its results\./);
});

test("team autocomplete filters by name and returns ids", async () => {
  const { json } = await call({ type: 4, data: { name: "odds", options: [opt("team", "ball", { focused: true })] } });
  assert.equal(json.type, 8);
  assert.deepEqual(json.data.choices, [{ name: "Ball Knowers", value: "3" }, { name: "Ballers Club", value: "4" }]);
  const all = await call({ type: 4, data: { name: "odds", options: [opt("team", "", { focused: true })] } });
  assert.equal(all.json.data.choices.length, 4);
});

test("/standings, /forecast and /matchups show the morning's tables", async () => {
  const st = (await call(command("standings"))).json.data.embeds[0];
  assert.equal(st.title, "Test League: standings");
  assert.equal(st.description, snapshot.text.standings);
  assert.ok(st.description.includes(" 1. Inevitable       1-0   0-1  +1  1.1k"));     // supposed to be 0-1
  assert.ok(st.description.includes("😬 **Not\\_Eating\\*Poop** (70% to win) lost to **Inevitable**"));
  const fc = (await call(command("forecast"))).json.data.embeds[0];
  assert.equal(fc.title, "Test League: projected final standings");
  assert.equal(fc.description, snapshot.text.forecast);
  const mu = (await call(command("matchups"))).json.data.embeds[0];
  assert.equal(mu.title, "Week 2 (Nov 2–8): matchups");
  assert.equal(mu.description, snapshot.text.matchups["2"]);
  const past = (await call(command("matchups", [opt("week", 1)]))).json.data.embeds[0];
  assert.equal(past.title, "Week 1 (Oct 26–Nov 1): results");
  assert.equal(past.description, snapshot.text.results["1"]);
  assert.match((await call(command("matchups", [opt("week", 7)]))).json.data.content, /No matchups for week 7/);
});

test("if the data can't be loaded, the user hears about it instead of an error", async () => {
  const e = { DISCORD_PUBLIC_KEY: publicHex, SNAPSHOT_URL: "https://data.test/broken.json" };
  const { json } = await call(command("standings"), { e });
  assert.equal(json.data.flags, 64);
  assert.match(json.data.content, /couldn't load the projections/);
  const ac = await call({ type: 4, data: { name: "odds", options: [opt("team", "in", { focused: true })] } }, { e });
  assert.deepEqual(ac.json.data, { choices: [] });
});

test("command definitions follow Discord's rules", () => {
  for (const c of COMMANDS) {
    assert.match(c.name, /^[a-z0-9_-]{1,32}$/);
    assert.ok(c.description.length >= 1 && c.description.length <= 100);
    for (const o of c.options || []) {
      assert.match(o.name, /^[a-z0-9_-]{1,32}$/);
      assert.ok(o.description.length <= 100);
    }
  }
});

test("percentages round like the Python side", () => {
  assert.deepEqual([0.001, 0.405, 0.415, 0.4, 0.996, null].map(pct), ["<1%", "40%", "42%", "40%", ">99%", "–"]);
});
