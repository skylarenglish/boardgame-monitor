import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import copy
from openskill.models import PlackettLuce
import streamlit as st
import networkx as nx
from itertools import combinations, chain
st.set_page_config()

def hierarchy_pos(G, root, levels=None, width=1., height=1.):
    '''If there is a cycle that is reachable from root, then this will see infinite recursion.
       G: the graph
       root: the root node
       levels: a dictionary
               key: level number (starting from 0)
               value: number of nodes in this level
       width: horizontal space allocated for drawing
       height: vertical space allocated for drawing'''
    TOTAL = "total"
    CURRENT = "current"
    def make_levels(levels, node=root, currentLevel=0, parent=None):
        """Compute the number of nodes for each level
        """
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


games_df = pd.read_csv('game_reports_cumulative.csv', parse_dates=[2,3])
games_df['players'] = games_df['players'].apply(lambda x: x.split(" "))
games_df['year'] = games_df['game_start'].dt.year
games_df['player_count'] = games_df['players'].apply(lambda x: len(x))
games_df['game_length'] = games_df['game_end'] - games_df['game_start']
games_df['game_length_minutes'] = games_df['game_length'].dt.total_seconds()/60
games_df.sort_values(['game_start','game_played'], inplace=True)
games_df.reset_index(inplace=True)


player_games_df = games_df.explode('players')
#player_games_df.sort_values(['game_played', 'game_start'], inplace=True)
player_games_df['place'] = player_games_df.groupby(['game_start', 'game_played'])['players'].expanding().count().values
#t * (1+((1-p)/(2n-2))) where t = time, p = place order, and n = # of players
t = player_games_df['game_length_minutes']
p = player_games_df['place']
n = player_games_df['player_count']
player_games_df['score'] = np.round(t * (1+((1-p)/(2*n-2))))
st.dataframe(player_games_df, column_config={'game_length':st.column_config.DatetimeColumn('game length')})


model = PlackettLuce()

player_rankings = {player:model.rating(name=player) for player in games_df['players'].explode().drop_duplicates().tolist()}

ranking_hist = []
for game in games_df['players'].to_list():
    match = [(lambda x: [x])(ranking) for ranking in [player_rankings.get(key) for key in game]]
    results = model.rate(match)
    ranking_hist.append(copy.deepcopy(player_rankings))
ranking_hist = pd.DataFrame(ranking_hist)

ranking_final = [(player, round(rating.mu,2), round(rating.sigma,2), round(rating.ordinal(),2)) for player, rating in sorted(player_rankings.items(), key=lambda x:x[1].ordinal(), reverse=True)]


player = st.selectbox('Select Player:',np.unique(player_games_df['players']))
st.write(f"{player}'s games:")
st.write(games_df[games_df['players'].apply(lambda x: player in x)])


diffs = games_df[games_df['players'].apply(lambda x: player in x)].join(ranking_hist[player].apply(lambda x: x.mu))[player].diff()
diffs.iloc[0] = games_df[games_df['players'].apply(lambda x: player in x)].join(ranking_hist[player].apply(lambda x: x.mu)).iloc[0][player] - 25
diffs = pd.DataFrame(diffs)
diffs['game'] = games_df[games_df['players'].apply(lambda x: player in x)]['game_played']
diffs.set_index('game', inplace=True)
diffs.plot(kind='bar')
st.write(f"{player}'s skill changes")
st.write(diffs)
st.bar_chart(diffs,width=1000,use_container_width=False)

array_of_unique_players = np.unique(player_games_df['players'])
win_matrix = np.empty((array_of_unique_players.size, array_of_unique_players.size))
for player_1 in array_of_unique_players:
    for player_2 in array_of_unique_players:
        win_matrix[np.argmax(array_of_unique_players == player_1), np.argmax(array_of_unique_players== player_2)] = round(model.predict_win([[player_rankings[player_1]], [player_rankings[player_2]]])[0],2)
st.write('Predicted Win Probability by Player')
st.write(pd.DataFrame(win_matrix, columns=array_of_unique_players, index=array_of_unique_players).loc[player].sort_values())

G = nx.Graph()
G.add_nodes_from(np.unique(player_games_df['players'].values))
G.add_edges_from(set(chain(*[list(combinations(game,2)) for game in [sorted(game) for game in games_df['players'].to_list()]])))

pos = hierarchy_pos(nx.bfs_tree(G, player),player)    
fig = plt.figure(2, figsize=(20,18), dpi=60)
nx.draw_networkx(G, pos=pos, with_labels=True, node_size=2000)
st.write(f'Who {player} has played')
st.pyplot(fig)