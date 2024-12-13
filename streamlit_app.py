import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import copy
from openskill.models import PlackettLuce
import streamlit as st

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
player_games_df


model = PlackettLuce()

player_rankings = {player:model.rating(name=player) for player in games_df['players'].explode().drop_duplicates().tolist()}

ranking_hist = []
for game in games_df['players'].to_list():
    match = [(lambda x: [x])(ranking) for ranking in [player_rankings.get(key) for key in game]]
    results = model.rate(match)
    print(player_rankings)
    ranking_hist.append(copy.deepcopy(player_rankings))

ranking_final = [(player, round(rating.mu,2), round(rating.sigma,2), round(rating.ordinal(),2)) for player, rating in sorted(player_rankings.items(), key=lambda x:x[1].ordinal(), reverse=True)]

st.write(games_df[games_df['players'].apply(lambda x: 'Max' in x)])

player = 'Skylar'
diffs = games_df[games_df['players'].apply(lambda x: player in x)].join(ranking_hist[player].apply(lambda x: x.mu))[player].diff()
diffs.iloc[0] = games_df[games_df['players'].apply(lambda x: player in x)].join(ranking_hist[player].apply(lambda x: x.mu)).iloc[0][player] - 25
diffs = pd.DataFrame(diffs)
diffs['game'] = games_df[games_df['players'].apply(lambda x: player in x)]['game_played']
diffs.set_index('game', inplace=True)
diffs.plot(kind='bar')
st.bar(diffs)
