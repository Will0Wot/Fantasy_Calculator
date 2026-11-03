# Discord bot

**What it does**

- **Weekly post.** On the first morning of each fantasy week (Mondays), it posts in your channel: last week's
  results (upsets marked), the standings next to the record each team was supposed to have, the projected final
  standings (with who will be eating poop), and this week's matchups with each team's chance to win.
- **Ask it anytime:**
  - `/odds team opponent [week]`: *"**Inevitable** has a **48%** chance of beating **Where my hug at fc** in
    Week 3 (Nov 2–8)."* Any two teams, any week left. If they don't actually play that week, it says it's a
    what-if.
  - `/standings`: the standings so far next to the record each team was *supposed* to have: a win for every
    week the model favored it before the week started. "Expected 2-0, actual 1-1" means it lost a week it was
    favored in; those weeks are listed for the last week played.
  - `/forecast`: the projected final standings: wins, and each team's chance to win the league or finish last.
  - `/matchups [week]`: a week's matchups with win chances, or its results once it's over.

The numbers are the app's own: the same projections, the same win-chance formula, refreshed every morning.

**How it fits together**

```
GitHub job, every morning ──► discord-bot/data/snapshot.json (committed) ──► Cloudflare Worker ──► answers /odds ...
GitHub job, start of each fantasy week ──► channel webhook ──► the weekly post
```

`tradecalc discord` (in `tradecalc/discord_bot.py`) builds the snapshot: the standings with expected records,
the league forecast, each team's projected
score for every week left, and the chance that every team beats every other team, every week. The Worker in
`src/` only looks things up, so it answers instantly and never needs the league's data or passwords.

## Setup (once, about 15 minutes)

You'll need: permission to manage your league's Discord server, a free Cloudflare account, and Node.js
(already on this Mac).

### 1. Create the Discord app

1. Go to <https://discord.com/developers/applications> → **New Application**, and name it (e.g. *Fantasy Odds*).
2. **General Information:** copy the **Application ID** and the **Public Key**.
3. **Bot** (left menu) → **Reset Token** → copy the token. It works like a password: it only goes in
   `discord-bot/.env` on your computer (step 4), never in chat or in the repo.
4. **OAuth2** → **URL Generator** → tick **applications.commands** → open the generated link → pick your league's
   server → **Authorize**.

### 2. The weekly post

1. In Discord: **Server Settings → Integrations → Webhooks → New Webhook**. Pick the channel, then **Copy Webhook URL**.
2. On GitHub: the repository → **Settings → Secrets and variables → Actions → New repository secret**.
   Name: `DISCORD_WEBHOOK_URL`, secret: the URL.
3. To try it now: **Actions → Season tracker → Run workflow**, tick *Post this week's Discord preview now*, then
   **Run workflow**. The post shows up when the job finishes (about 8 minutes).

### 3. Put the Worker online

In Terminal, from the project folder:

```bash
cd discord-bot
```

```bash
npm install
```

```bash
npx wrangler login
```

This opens your browser to sign in to Cloudflare, or to create the free account.

```bash
npm run deploy
```

This prints the Worker's address, like `https://fantasy-odds-bot.<your-name>.workers.dev`.

```bash
npx wrangler secret put DISCORD_PUBLIC_KEY
```

Paste the **Public Key** from step 1.2.

Then, in the Developer Portal → **General Information → Interactions Endpoint URL**: paste the Worker's
address → **Save Changes**. Discord tests it on the spot. If it says the URL can't be verified, the public key
is missing or wrong. Run the last command again.

### 4. Register the commands

Create a file named `.env` in `discord-bot/` (git ignores it) with:

```
DISCORD_TOKEN=the bot token from step 1.3
DISCORD_APPLICATION_ID=the application id from step 1.2
DISCORD_GUILD_ID=your server's id
```

`DISCORD_GUILD_ID` is optional but makes the commands appear right away. To get it, turn on Discord's
**User Settings → Advanced → Developer Mode**, then right-click the server's icon → **Copy Server ID**. Then:

```bash
npm run register
```

Type `/odds` in your server to try it.

## Changing things

| To change | Edit | Then |
|---|---|---|
| The weekly post or the tables | `tradecalc/discord_bot.py` | nothing: the next morning's run uses it |
| A link to the app in the weekly post | `discord.app_url` in `tradecalc/config/league.yaml` | nothing |
| How the bot answers | `src/bot.js` | `npm test`, then `npm run deploy` |
| The commands themselves | `src/commands.js` | `npm run register` |

To preview the weekly post without sending it: `tradecalc discord --preview`.

## Costs and safety

- **Costs:** free. Cloudflare's free plan allows 100,000 requests a day, and GitHub Actions is free for public
  repositories.
- **Secrets:** the bot token, public key and webhook URL never go in the repository. They live in `.env` (on your
  computer), a Cloudflare secret and a GitHub secret.
- **Webhook URL:** anyone who has it can post in that channel. If it leaks, delete the webhook in Discord and make
  a new one.
- **The snapshot is public,** like the repository: team names, records and projections, the same as the app shows.
