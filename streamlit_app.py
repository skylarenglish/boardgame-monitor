import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import copy
from openskill.models import PlackettLuce
import streamlit as st
import networkx as nx
from itertools import combinations, chain
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(page_title="Boardgame Monitor", layout="wide")

# --- Helper Functions ---
def hierarchy_pos(G, root, levels=None, width=1., height=1.):
    TOTAL = "total"
    CURRENT = "current"
    def make_levels(levels, node=root, currentLevel=0, parent=None):
        if not currentLevel in levels:
            levels[currentLevel] = {TOTAL : 0, CURRENT : 0}
        levels[currentLevel][TOTAL] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                levels =  make_levels(levels, neighbor, currentLevel + 1, node)
        return levels

    def make_pos(pos, node=root, currentLevel=0, parent=None, vert_loc=0):
        dx = 1/levels[currentLevel][TOTAL]
        left = dx/2
        pos[node] = ((left + dx*levels[currentLevel][CURRENT])*width, vert_loc)
        levels[currentLevel][CURRENT] += 1
        neighbors = G.neighbors(node)
        for neighbor in neighbors:
            if not neighbor == parent:
                pos = make_pos(pos, neighbor, currentLevel + 1, node, vert_loc-vert_gap)
        return pos
    if levels is None:
        levels = make_levels({})
    else:
        levels = {l:{TOTAL: levels[l], CURRENT:0} for l in levels}
    vert_gap = height / (max([l for l in levels])+1)
    return make_pos({})

@st.cache_data(ttl=60) # Cache for 1 minute
def load_data():
    df = pd.read_csv('game_reports_cumulative.csv', parse_dates=[2,3])
    # Ensure standard types
    df['game_start'] = pd.to_datetime(df['game_start'])
    df['game_end'] = pd.to_datetime(df['game_end'])
    
    # Process players
    df['players_list'] = df['players'].apply(lambda x: x.split(" "))
    df['year'] = df['game_start'].dt.year
    df['player_count'] = df['players_list'].apply(lambda x: len(x))
    df['game_length'] = df['game_end'] - df['game_start']
    df['game_length_minutes'] = df['game_length'].dt.total_seconds()/60
    df.sort_values(['game_start','game_played'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def get_rankings(df):
    model = PlackettLuce()
    all_players = sorted(list(set(chain.from_iterable(df['players_list']))))
    player_rankings = {player: model.rating(name=player) for player in all_players}
    
    ranking_hist = []
    for _, row in df.iterrows():
        game_players = row['players_list']
        # Match expected by openskill: [[rating1], [rating2], ...]
        match = [[player_rankings[p]] for player in game_players for p in [player]]
        # Match is actually a list of lists of ratings
        match = [[player_rankings[p]] for p in game_players]
        player_rankings_updated = model.rate(match)
        
        # Update local dict
        for i, p in enumerate(game_players):
            player_rankings[p] = player_rankings_updated[i][0]
            
        ranking_hist.append(copy.deepcopy(player_rankings))
        
    hist_df = pd.DataFrame(ranking_hist)
    
    # Final rankings list
    ranking_final = []
    for p in all_players:
        r = player_rankings[p]
        ranking_final.append({
            'Player': p,
            'Mu': round(r.mu, 2),
            'Sigma': round(r.sigma, 2),
            'Skill': round(r.ordinal(), 2)
        })
    
    return pd.DataFrame(ranking_final).sort_values('Skill', ascending=False), hist_df, player_rankings, model

# --- Navigation ---
st.sidebar.title("🎲 Game Monitor")
page = st.sidebar.radio("Go to", ["Leaderboard (TV Mode)", "Submit Result", "Detailed Stats"])

# Load data and calc once
games_df = load_data()
rank_df, ranking_hist, final_ratings, model = get_rankings(games_df)

if page == "Leaderboard (TV Mode)":
    st.markdown("<h1 style='text-align: center; font-size: 60px;'>🏆 Current Leaderboard</h1>", unsafe_markdown_html=True)
    
    cols = st.columns([1, 2, 1])
    with cols[1]:
        # Big Leaderboard Table
        display_df = rank_df[['Player', 'Skill']].head(10).reset_index(drop=True)
        display_df.index += 1
        st.table(display_df)
        
    st.markdown("---")
    st.subheader("Latest Games")
    recent = games_df.tail(5)[['game_played', 'players', 'game_end']].sort_values('game_end', ascending=False)
    st.dataframe(recent, use_container_width=True)
    
    # Auto-refresh logic (meta tag refresh)
    st.markdown('<meta http-equiv="refresh" content="60">', unsafe_allow_html=True)
    st.info("Refreshing every 60 seconds...")

elif page == "Submit Result":
    st.title("➕ Submit Game Result")
    
    with st.form("game_form", clear_on_submit=True):
        game_name = st.text_input("Game Played", placeholder="e.g. Dune Imperium")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now())
            start_time = st.time_input("Start Time", datetime.now() - timedelta(hours=1))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
            end_time = st.time_input("End Time", datetime.now())
            
        all_known_players = sorted(list(set(chain.from_iterable(games_df['players_list']))))
        selected_players = st.multiselect("Who played? (In order of finish: 1st, 2nd, 3rd...)", all_known_players)
        
        new_players = st.text_input("New Players? (Space separated names)", help="If someone isn't in the list above, type their names here.")
        
        submitted = st.form_submit_button("Submit Result")
        
        if submitted:
            if not game_name or (not selected_players and not new_players):
                st.error("Please provide a game name and at least one player.")
            else:
                # Combine players
                final_player_order = selected_players
                if new_players:
                    final_player_order += new_players.split(" ")
                
                # Format strings
                players_str = " ".join(final_player_order)
                start_dt = datetime.combine(start_date, start_time).strftime("%Y-%m-%d %H:%M")
                end_dt = datetime.combine(end_date, end_time).strftime("%Y-%m-%d %H:%M")
                
                # Append to CSV
                new_row = f"\n{game_name},{players_str},{start_dt},{end_dt}"
                with open('game_reports_cumulative.csv', 'a') as f:
                    f.write(new_row)
                
                st.success(f"Successfully recorded {game_name}! Leaderboard updated.")
                st.balloons()

elif page == "Detailed Stats":
    st.title("📊 Detailed Group Analytics")
    
    tabs = st.tabs(["Player Focus", "Win Probabilities", "Social Graph", "Raw Data"])
    
    with tabs[0]:
        player = st.selectbox('Select Player:', rank_df['Player'].unique())
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            p_rank = rank_df[rank_df['Player'] == player].iloc[0]
            st.metric("Current Skill (Ordinal)", p_rank['Skill'])
            st.metric("Mu (Mean)", p_rank['Mu'])
            st.metric("Sigma (Uncertainty)", p_rank['Sigma'])
            
        with col2:
            st.write(f"### {player}'s Skill History")
            player_mu_history = ranking_hist[player].apply(lambda x: x.mu)
            st.line_chart(player_mu_history)

        st.write(f"### {player}'s Recent Games")
        st.write(games_df[games_df['players_list'].apply(lambda x: player in x)].tail(10))

    with tabs[1]:
        st.header("Predicted Win Probability")
        st.write("Chance of Player A (row) beating Player B (column)")
        
        array_of_unique_players = rank_df['Player'].unique()
        win_matrix = np.empty((array_of_unique_players.size, array_of_unique_players.size))
        
        for i, p1 in enumerate(array_of_unique_players):
            for j, p2 in enumerate(array_of_unique_players):
                if p1 == p2:
                    win_matrix[i, j] = 0.5
                else:
                    win_matrix[i, j] = round(model.predict_win([[final_ratings[p1]], [final_ratings[p2]]])[0], 2)
        
        wm_df = pd.DataFrame(win_matrix, columns=array_of_unique_players, index=array_of_unique_players)
        st.dataframe(wm_df.style.background_gradient(cmap='RdYlGn'))

    with tabs[2]:
        st.header("Social Hierarchy Graph")
        focus_player = st.selectbox('Focus Player for Graph:', rank_df['Player'].unique(), key="graph_player")
        
        G = nx.Graph()
        G.add_nodes_from(rank_df['Player'].values)
        G.add_edges_from(set(chain(*[list(combinations(game,2)) for game in [sorted(game) for game in games_df['players_list'].to_list()]])))

        try:
            pos = hierarchy_pos(nx.bfs_tree(G, focus_player), focus_player)    
            fig = plt.figure(figsize=(12, 8))
            nx.draw_networkx(G, pos=pos, with_labels=True, node_size=1500, node_color="skyblue", font_size=10)
            st.pyplot(fig)
        except Exception as e:
            st.error("Could not generate hierarchy graph. Group might not be fully connected.")

    with tabs[3]:
        st.header("Exploded Player Data")
        # Reproduce the player_games_df from original
        player_games_df = games_df.explode('players_list')
        player_games_df['place'] = player_games_df.groupby(['game_start', 'game_played']).cumcount() + 1
        
        st.dataframe(player_games_df)
