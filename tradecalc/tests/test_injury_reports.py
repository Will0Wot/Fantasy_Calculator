from tradecalc.ingest.injury_reports import TEAM_BY_SQUASHED, box_score_key, player_key, squash


def test_player_keys_match_box_scores_in_both_pdf_formats():
    # old PDFs keep spaces, newer ones (2025-26) drop them; suffixes and accents vary
    assert player_key("Carey Jr., Vernon") == player_key("CareyJr.,Vernon") == box_score_key("Vernon Carey Jr.")
    assert player_key("McConnell, T.J.") == box_score_key("T.J. McConnell")
    assert player_key("Jokic, Nikola") == box_score_key("Nikola Jokić")
    assert player_key("Porter Jr., Michael") == box_score_key("Michael Porter Jr.")
    assert player_key("Williams III, Robert") == box_score_key("Robert Williams III")


def test_team_names_map_with_or_without_spaces():
    assert TEAM_BY_SQUASHED[squash("IndianaPacers")] == "IND"
    assert TEAM_BY_SQUASHED[squash("Portland Trail Blazers")] == "POR"
    assert TEAM_BY_SQUASHED[squash("LA Clippers")] == TEAM_BY_SQUASHED[squash("Los Angeles Clippers")] == "LAC"
    assert len(set(TEAM_BY_SQUASHED.values())) == 30


def test_attach_played_matches_aliases_and_last_names():
    import pandas as pd

    from tradecalc.ingest.injury_reports import attach_played

    logs = pd.DataFrame({"player_name": ["Bub Carrington", "Nikola Jokić", "Jalen Williams", "Jaylin Williams"],
                         "team": ["WAS", "DEN", "OKC", "OKC"], "game_date": ["2025-01-02"] * 4, "min": [30, 35, 0, 12]})
    rep = pd.DataFrame({"player": ["Carrington,Carlton", "Jokic, Nikola", "Williams, J.", "Smith, Nobody"],
                        "team_abbr": ["WAS", "DEN", "OKC", "DEN"], "game_date": ["2025-01-02"] * 4})
    rep["player_key"] = rep.player.map(player_key)
    out = attach_played(rep, logs)
    assert list(out.played) == [True, True, False, False]
    assert pd.isna(out.box_key.iloc[2])          # two OKC Williamses: ambiguous, not guessed
