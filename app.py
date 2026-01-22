from __future__ import annotations

from pathlib import Path
import itertools
import pandas as pd
import streamlit as st

# ----------------------------
# Page config + light styling
# ----------------------------
st.set_page_config(page_title="Wheelchair Rugby Dashboard", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; }
      .card {
        padding: 1rem 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.03);
      }
      .subtle { opacity: 0.85; }
      .tiny { font-size: 0.9rem; opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_DIR = Path(__file__).resolve().parent
PLAYER_CSV = APP_DIR / "player_data.csv"
STINT_CSV = APP_DIR / "stint_data.csv"


# ----------------------------
# Data loading
# ----------------------------
@st.cache_data
def load_player_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect: player, rating
    df["team"] = df["player"].astype(str).str.split("_", n=1).str[0]
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
    return df


@st.cache_data
def load_stint_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expected columns: game_id,h_team,a_team,minutes,h_goals,a_goals,home1..home4,away1..away4
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
    df["h_goals"] = pd.to_numeric(df["h_goals"], errors="coerce").fillna(0.0)
    df["a_goals"] = pd.to_numeric(df["a_goals"], errors="coerce").fillna(0.0)

    home_cols = ["home1", "home2", "home3", "home4"]
    away_cols = ["away1", "away2", "away3", "away4"]

    # Precompute sorted lineup keys for quick matching
    df["home_key"] = df[home_cols].apply(lambda r: tuple(sorted(map(str, r.values))), axis=1)
    df["away_key"] = df[away_cols].apply(lambda r: tuple(sorted(map(str, r.values))), axis=1)
    return df


def safe_load() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Works both locally + (if you ever run it elsewhere) with fallbacks
    if not PLAYER_CSV.exists():
        st.error(f"Missing {PLAYER_CSV.name} in {APP_DIR}")
        st.stop()
    if not STINT_CSV.exists():
        st.error(f"Missing {STINT_CSV.name} in {APP_DIR}")
        st.stop()

    p = load_player_data(PLAYER_CSV)
    s = load_stint_data(STINT_CSV)
    return p, s


players_df, stints_df = safe_load()


# ----------------------------
# Session state: roster status
# ----------------------------
def init_state_for_team(team: str, team_players: list[str]) -> None:
    """
    Keep one shared roster state for the chosen team,
    used by BOTH Home and Away sections.
    """
    key = f"roster_status__{team}"
    if key not in st.session_state:
        # default: everyone active
        st.session_state[key] = {p: "Active" for p in team_players}


def get_status_map(team: str) -> dict[str, str]:
    return st.session_state[f"roster_status__{team}"]


def set_status(team: str, names: list[str], new_status: str) -> None:
    m = get_status_map(team)
    for n in names:
        if n in m:
            m[n] = new_status


# ----------------------------
# Analytics helpers
# ----------------------------
def lineup_rating_sum(lineup: tuple[str, ...], player_df_team: pd.DataFrame) -> float:
    sub = player_df_team[player_df_team["player"].isin(lineup)][["player", "rating"]]
    return float(sub["rating"].sum())


def lineup_stint_stats(
    team: str,
    context: str,  # "home" or "away"
    lineup: tuple[str, ...],
    stints: pd.DataFrame,
) -> dict:
    """
    Returns summary stats for a given lineup when team is home/away.
    Matches stints by set equality (order-insensitive).
    """
    key = tuple(sorted(lineup))

    if context == "home":
        df = stints[stints["h_team"] == team]
        df = df[df["home_key"] == key]
        goals_for = df["h_goals"].sum()
        goals_against = df["a_goals"].sum()
    else:
        df = stints[stints["a_team"] == team]
        df = df[df["away_key"] == key]
        goals_for = df["a_goals"].sum()
        goals_against = df["h_goals"].sum()

    minutes = df["minutes"].sum()
    gd = goals_for - goals_against
    gd_per_min = (gd / minutes) if minutes > 0 else 0.0

    return {
        "matches": len(df),
        "minutes": float(minutes),
        "goals_for": float(goals_for),
        "goals_against": float(goals_against),
        "goal_diff": float(gd),
        "goal_diff_per_min": float(gd_per_min),
        "stints_df": df,
    }


def find_best_lineup(
    team: str,
    context: str,  # "home" or "away"
    active_players: list[str],
    player_df_team: pd.DataFrame,
    stints: pd.DataFrame,
    rating_cap: float,
) -> dict | None:
    """
    Placeholder optimizer:
    brute-force all 4-player combinations that satisfy rating_cap,
    score by goal_diff_per_min (from stints).
    """
    if len(active_players) < 4:
        return None

    best = None
    for lineup in itertools.combinations(active_players, 4):
        rsum = lineup_rating_sum(lineup, player_df_team)
        if rsum > rating_cap:
            continue

        stats = lineup_stint_stats(team, context, lineup, stints)
        score = stats["goal_diff_per_min"]

        candidate = {
            "lineup": lineup,
            "rating_sum": rsum,
            "score": score,
            **stats,
        }

        if best is None or candidate["score"] > best["score"]:
            best = candidate

    return best


# ----------------------------
# Header
# ----------------------------
st.title("🏉 Wheelchair Rugby — Lineups & Stints Dashboard")
st.caption("WCR Analytics Deepdive")

with st.expander("📁 Expected files in this folder (you already have these)", expanded=False):
    st.code(
        "CaseStudy1/\n"
        "  app.py\n"
        "  player_data.csv\n"
        "  stint_data.csv\n"
        "  Wheelchair_Rugby_EDA.ipynb\n"
        "  venv/\n",
        language="text",
    )

# ----------------------------
# Sidebar controls
# ----------------------------
teams = sorted(set(stints_df["h_team"]).union(set(stints_df["a_team"])))
default_team = "Canada" if "Canada" in teams else (teams[0] if teams else None)

st.sidebar.header("⚙️ Controls")
team = st.sidebar.selectbox("Select team", teams, index=teams.index(default_team) if default_team in teams else 0)

team_players_df = players_df[players_df["team"] == team].copy()
team_players = sorted(team_players_df["player"].tolist())
init_state_for_team(team, team_players)

rating_cap = st.sidebar.slider("Lineup rating cap (placeholder constraint)", 4.0, 12.0, 8.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("**What’s live right now:**")
st.sidebar.markdown("- Roster moves (Active ↔ Injured)\n- Lineup selector + stint stats\n- Home vs Away analytics")

# shared roster state for both sections
status_map = get_status_map(team)
active = sorted([p for p, s in status_map.items() if s == "Active"])
injured = sorted([p for p, s in status_map.items() if s == "Injured"])


# ----------------------------
# Top summary row
# ----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Team", team)
c2.metric("Active", len(active))
c3.metric("Injured", len(injured))
c4.metric("Total Players", len(team_players))

st.markdown("---")

# ----------------------------
# Layout: Home & Away sections
# ----------------------------
tab_roster, tab_lineups, tab_explore = st.tabs(["🧑‍🤝‍🧑 Roster Manager", "🏆 Best Lineups", "🔎 Stint Explorer"])

with tab_roster:
    st.subheader("Roster Manager (shared across Home & Away views)")
    st.write("Move players between **Active** and **Injured**. This updates instantly and affects the lineup search.")

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ✅ Active Players")
        pick_active = st.multiselect("Select Active to mark Injured", active, key=f"pick_active_{team}")
        move1 = st.button("➡️ Move to Injured", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🩹 Injured Players")
        pick_inj = st.multiselect("Select Injured to mark Active", injured, key=f"pick_inj_{team}")
        move2 = st.button("⬅️ Move to Active", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if move1 and pick_active:
        set_status(team, pick_active, "Injured")
        st.rerun()

    if move2 and pick_inj:
        set_status(team, pick_inj, "Active")
        st.rerun()

    st.markdown("### Ratings snapshot")
    # quick ratings chart
    ratings_view = team_players_df.copy()
    ratings_view["status"] = ratings_view["player"].map(status_map)
    ratings_view = ratings_view.sort_values(["status", "rating"], ascending=[True, False])
    st.bar_chart(ratings_view.set_index("player")["rating"])


with tab_lineups:
    st.subheader("Home Games vs Away Games — Lineup Explorer")
    st.caption(
        "Pick any 4-player lineup to see stint-based performance metrics. "
        "Home and away are tracked separately."
    )

    home_col, away_col = st.columns(2)

    def render_best_lineup(context_label: str, context_key: str):
        # context_key is "home" or "away"
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"## {context_label}")

        top1, top2, top3 = st.columns(3)
        top1.metric("Active available", len(active))
        top2.metric("Rating cap", f"{rating_cap:.1f}")
        # games count in this context
        if context_key == "home":
            gcount = stints_df[stints_df["h_team"] == team]["game_id"].nunique()
        else:
            gcount = stints_df[stints_df["a_team"] == team]["game_id"].nunique()
        top3.metric("Games in data", int(gcount))

        lineup_key = f"lineup_select_{team}_{context_key}"
        current_lineup = st.session_state.get(lineup_key, [])
        filtered_lineup = [p for p in current_lineup if p in active]
        if filtered_lineup != current_lineup:
            st.session_state[lineup_key] = filtered_lineup

        selected_lineup = st.multiselect(
            "Lineup (pick 4 active players)",
            active,
            key=lineup_key,
            help="Build a lineup manually.",
        )

        if len(selected_lineup) != 4:
            st.info("Select 4 players to view lineup stats.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        lineup = tuple(selected_lineup)
        rating_sum = lineup_rating_sum(lineup, team_players_df)
        stats = lineup_stint_stats(team, context_key, lineup, stints_df)

        if rating_sum > rating_cap:
            st.warning(f"Lineup rating total ({rating_sum:.1f}) exceeds the cap ({rating_cap:.1f}).")

        st.markdown("### ✅ Lineup")
        lineup_table = (
            team_players_df[team_players_df["player"].isin(lineup)]
            .sort_values("rating", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(lineup_table, use_container_width=True, hide_index=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rating total", f"{rating_sum:.1f}")
        m2.metric("Matched stints", int(stats["matches"]))
        m3.metric("Minutes", f"{stats['minutes']:.2f}")
        m4.metric("Goal diff / min", f"{stats['goal_diff_per_min']:.3f}")

        # quick “stint stats” block
        st.markdown("### 📊 Stint summary for this lineup")
        s1, s2, s3 = st.columns(3)
        s1.metric("Goals for", int(stats["goals_for"]))
        s2.metric("Goals against", int(stats["goals_against"]))
        s3.metric("Goal diff", int(stats["goal_diff"]))

        # simple performance chart (single bar)
        perf = pd.DataFrame(
            [{"metric": "Goal diff per min", "value": stats["goal_diff_per_min"]}]
        ).set_index("metric")
        st.bar_chart(perf)

        # show raw matched stints (useful for debugging)
        with st.expander("Show matched stints (raw rows)", expanded=False):
            show_cols = [
                "game_id", "h_team", "a_team", "minutes", "h_goals", "a_goals",
                "home1", "home2", "home3", "home4", "away1", "away2", "away3", "away4",
            ]
            st.dataframe(stats["stints_df"][show_cols].reset_index(drop=True), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with home_col:
        render_best_lineup("🏠 Home games", "home")
    with away_col:
        render_best_lineup("✈️ Away games", "away")

    st.markdown("---")
    st.markdown("### 📌 Next step (when your colleague’s optimizer is ready)")
    st.info("Add a ranked recommendations panel here if you want to surface auto-generated lineups.")


with tab_explore:
    st.subheader("Stint Explorer (fast sanity checks + plots)")

    c1, c2 = st.columns([1, 1])
    with c1:
        context = st.selectbox("Context", ["Home", "Away"])
    with c2:
        min_minutes = st.slider("Minimum stint minutes", 0.0, 10.0, 0.0, 0.25)

    if context == "Home":
        df = stints_df[stints_df["h_team"] == team].copy()
        df["gd"] = df["h_goals"] - df["a_goals"]
    else:
        df = stints_df[stints_df["a_team"] == team].copy()
        df["gd"] = df["a_goals"] - df["h_goals"]

    df = df[df["minutes"] >= min_minutes]

    s1, s2, s3 = st.columns(3)
    s1.metric("Stints", len(df))
    s2.metric("Total minutes", f"{df['minutes'].sum():.2f}")
    s3.metric("Avg goal diff / stint", f"{df['gd'].mean():.2f}" if len(df) else "—")

    st.markdown("### Quick chart: goal diff distribution (by stint)")
    chart_df = df[["gd"]].copy()
    # Streamlit's built-in charting is simple but reliable
    st.bar_chart(chart_df["gd"].value_counts().sort_index())

    with st.expander("Raw stints (filtered)", expanded=False):
        cols = [
            "game_id", "h_team", "a_team", "minutes", "h_goals", "a_goals",
            "home1", "home2", "home3", "home4", "away1", "away2", "away3", "away4"
        ]
        st.dataframe(df[cols].reset_index(drop=True), use_container_width=True)
