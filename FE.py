import pandas as pd
import difflib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import streamlit as st


def stm_user_data(us_id):
    user_data = pd.read_csv('C:/Users/desti/PycharmProjects/App/steam-200k.csv',
                            names=['id', 'game', 'type', 'hours', '0'])
    user_data = user_data[user_data["type"].str.contains("play")]
    user_data.drop(['type', '0'], axis=1, inplace=True)
    user_id = us_id
    dummy = user_data[user_data.id == user_id]
    dummy.sort_values(by=['hours'], ascending=False)
    dummy.reset_index(inplace=True, drop=True)
    user_preferences = dummy.head(3).game
    return user_preferences


cv = CountVectorizer(max_features=370, stop_words='english')
df = pd.read_csv('C:/Users/desti/PycharmProjects/App/steam.csv')
df.insert(loc=0, column="Index", value=df.index)
gms = df[['title', 'platforms', 'steamspy_tags']]
vct = cv.fit_transform(gms['steamspy_tags']).toarray()
close_vct = cosine_similarity(vct)

recm_list = list()


def gms_sys(game):
    gm_ind = gms[gms['title'] == game].index[0]
    sim_dis = close_vct[gm_ind]
    game_list = sorted(list(enumerate(sim_dis)), reverse=True, key=lambda x: x[1])[1:10]

    for i in game_list:
        rmd_gm = gms.iloc[i[0]].name
        if rmd_gm not in recm_list:
            recm_list.append(rmd_gm)


def gm_lis(id):
    list_of_games = gms['title'].tolist()
    do_not_recommend_list = list()
    user_list = stm_user_data(id)
    for game in user_list:
        find_close_match = difflib.get_close_matches(game, list_of_games)
        close_match = find_close_match[0]
        do_not_recommend_list.append(close_match)
        gms_sys(close_match)

    recm_df = pd.merge(pd.DataFrame(recm_list, columns=['Index']), df, how='inner', on='Index')
    recm_df.sort_values(by='positive_ratings', ascending=False, inplace=True)
    recm_df = recm_df['title']
    return recm_df, do_not_recommend_list


st.title("GMS")
st.header("Game Recommendation System")
stm_id = st.number_input("Enter Your Steam ID:", min_value=1, max_value=151603713, value=1, step=1)
if st.button('Recommend Games'):
    game_rec, no_game = gm_lis(stm_id)
    j = 1
    for g in game_rec:
        if g not in no_game:
            if j < 11:
                st.write(g)
                j += 1
            else:
                break
