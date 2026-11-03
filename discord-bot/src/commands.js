// Slash commands, registered with Discord by `npm run register` (register.mjs).
const WEEK = {
  type: 4, name: "week", description: "Fantasy week number (default: this week)", required: false,
  min_value: 1, max_value: 30,
};

export const COMMANDS = [
  {
    name: "odds",
    description: "Chance one team beats another, this week or any week left",
    options: [
      { type: 3, name: "team", description: "A team", required: true, autocomplete: true },
      { type: 3, name: "opponent", description: "The team they'd face", required: true, autocomplete: true },
      WEEK,
    ],
  },
  { name: "standings", description: "The standings so far, with each team's expected wins and luck" },
  { name: "forecast", description: "Projected final standings: wins, chance to win the league or finish last" },
  { name: "matchups", description: "A week's matchups with each team's chance to win", options: [WEEK] },
];
