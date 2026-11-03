// Cloudflare Worker: Discord's interactions endpoint for the fantasy odds bot. Discord POSTs every slash
// command here; each request's Ed25519 signature is checked against the app's public key (Discord also
// tests that bad signatures are refused). Answers come from the snapshot the daily GitHub job commits
// (SNAPSHOT_URL), kept for two minutes per Worker instance.
import { handleInteraction } from "./bot.js";

const FRESH_MS = 2 * 60 * 1000;
const snapshots = new Map(); // url -> { at, data }
const keys = new Map(); // public key hex -> CryptoKey

async function loadSnapshot(url) {
  const hit = snapshots.get(url);
  if (hit && Date.now() - hit.at < FRESH_MS) return hit.data;
  const r = await fetch(url, { cf: { cacheTtl: 120, cacheEverything: true } });
  if (!r.ok) throw new Error(`snapshot: HTTP ${r.status}`);
  const data = await r.json();
  snapshots.set(url, { at: Date.now(), data });
  return data;
}

function hexToBytes(hex) {
  if (!/^(?:[0-9a-fA-F]{2})+$/.test(hex)) throw new Error("not hex");
  return Uint8Array.from(hex.match(/../g), (b) => parseInt(b, 16));
}

async function publicKey(hex) {
  if (!keys.has(hex)) {
    const raw = hexToBytes(hex);
    let key;
    try {
      key = await crypto.subtle.importKey("raw", raw, { name: "Ed25519" }, false, ["verify"]);
    } catch {
      // Cloudflare's legacy name for the same algorithm
      key = await crypto.subtle.importKey("raw", raw, { name: "NODE-ED25519", namedCurve: "NODE-ED25519" }, false, ["verify"]);
    }
    keys.set(hex, key);
  }
  return keys.get(hex);
}

async function signedByDiscord(request, body, publicKeyHex) {
  const signature = request.headers.get("X-Signature-Ed25519");
  const timestamp = request.headers.get("X-Signature-Timestamp");
  if (!signature || !timestamp || !publicKeyHex) return false;
  try {
    const key = await publicKey(publicKeyHex.trim());
    return await crypto.subtle.verify(key.algorithm, key, hexToBytes(signature), new TextEncoder().encode(timestamp + body));
  } catch {
    return false;
  }
}

export default {
  async fetch(request, env) {
    if (request.method === "GET") return new Response("Fantasy odds bot is running.");
    if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });
    const body = await request.text();
    if (!(await signedByDiscord(request, body, env.DISCORD_PUBLIC_KEY))) {
      return new Response("Bad request signature", { status: 401 });
    }
    let interaction;
    try {
      interaction = JSON.parse(body);
    } catch {
      return new Response("Bad request", { status: 400 });
    }
    return Response.json(await handleInteraction(interaction, () => loadSnapshot(env.SNAPSHOT_URL)));
  },
};
