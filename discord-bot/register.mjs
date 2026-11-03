// Registers the bot's slash commands with Discord. Run once, and again whenever src/commands.js changes:
//   npm run register
// Reads discord-bot/.env (never committed): DISCORD_TOKEN, DISCORD_APPLICATION_ID, and optionally
// DISCORD_GUILD_ID to register in one server instantly (global commands can take a while to show up).
import { COMMANDS } from "./src/commands.js";

const { DISCORD_TOKEN: token, DISCORD_APPLICATION_ID: appId, DISCORD_GUILD_ID: guildId } = process.env;
if (!token || !appId) {
  console.error("Missing DISCORD_TOKEN or DISCORD_APPLICATION_ID: put them in discord-bot/.env (see README.md).");
  process.exit(1);
}
const url = guildId
  ? `https://discord.com/api/v10/applications/${appId}/guilds/${guildId}/commands`
  : `https://discord.com/api/v10/applications/${appId}/commands`;
const response = await fetch(url, {
  method: "PUT",
  headers: { "Content-Type": "application/json", Authorization: `Bot ${token}` },
  body: JSON.stringify(COMMANDS),
});
if (!response.ok) {
  console.error(`Discord said ${response.status}: ${await response.text()}`);
  process.exit(1);
}
const done = await response.json();
console.log(`Registered ${done.map((c) => "/" + c.name).join(", ")} ${guildId ? "in your server" : "globally"}.`);
