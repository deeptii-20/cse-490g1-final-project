import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# stores user, track, and interaction data from passed in dataSet
class SpotifyDataset(Dataset):
    def __init__(self, dataSet):
        self.users = dataSet['user_id']
        self.tracks = dataSet['track_id']
        self.scores = dataSet['score']

    def __len__(self): # returns the number of unique (user, track) pairings
        return len(self.users)
  
    def __getitem__(self, idx):
        return self.users[idx], self.tracks[idx], self.scores[idx]

    def user_size(self):
        return len(self.users)

    def track_size(self):
        return len(self.tracks)

# Creates training and testing dataframes. Training data is 80% of randomly selected tracks per user and testing
# data is the remaining 20% of the tracks.
# @param file the file of spotify data
# @returns [train dataframe, test dataframe]. Dataframe has columns: user_id, track_id, score
def getTrainTestData(file):
    # get user and track matrices
    user_embed, user_ids = process_user_data(file)
    track_embed, track_ids = process_track_data(file)

    # calculate normalized scores per user for each track
    normalized_scores = get_scores(user_ids, track_ids, user_embed, track_embed)

    # get full lists of users and tracks
    all_users = []
    all_tracks = []
    for user in user_ids:
        for track in track_ids:
            all_users.append(user)
            all_tracks.append(track)
    
    # create dataframe of: user_id, track_id, scores
    d = {'user_id': all_users, 'track_id': all_tracks, 'score':normalized_scores.flatten()}
    data = pd.DataFrame(data=d)

    # get training and testing data
    train = []
    test = []
    for user_id in user_ids:
        # get all rows for a user
        user_rows = data.loc[data['user_id'] == user_id]
        
        # randomly split data into training and testing data (appending list faster than appending dataframes)
        curr_train = user_rows.sample(frac=0.8)
        for row in curr_train.to_numpy().tolist():
            train.append(row)

        curr_test = user_rows.drop(curr_train.index)
        for row in curr_test.to_numpy().tolist():
            test.append(row)
    colNames = ['user_id', 'track_id', 'score']
    return pd.DataFrame(train, columns=colNames), pd.DataFrame(test, columns=colNames)

def get_scores(user_ids, track_ids, user_embed, track_embed):
    # convert user dictionary to numpy array
    user_features = []
    for user in user_ids:
        user_features.append(user_embed[user].tolist())
    user_arr = np.array(user_features)

    # convert track dictionary to numpy array
    track_features = []
    for track in track_ids:
        track_features.append(track_embed[track].tolist())
    track_arr = np.array(track_features)

    # calculate potential user scores of each track and normalize it between 0 - 1
    scores = np.dot(user_arr, track_arr.T)
    normalized_scores = (scores-np.min(scores)) / (np.max(scores) - np.min(scores))
    return normalized_scores

def remove_from_list(list, value):
    c = list.count(value)
    for i in range(c):
        list.remove(value)

def get_feature_values(track):
    feature_labels = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence', 'tempo']
    feature_values = []
    for label in feature_labels:
        value = track[label]
        feature_values.append(value)
    return feature_values

def get_feature_weight(track):
    weight_labels = ['is_on_playlist', 'is_saved', 'is_recently_played', 'is_followed_artist']
    weight_values = [0.25, 0.25, 0.25, 0.25]
    feature_weights = []
    for label in weight_labels:
        weight = track[label]
        feature_weights.append(weight)
    return sum(i[0] * i[1] for i in zip(feature_weights, weight_values))

def process_user_data(file):
    colNames = ['user_id', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence', 'tempo', 'is_on_playlist', 'is_saved', 'is_recently_played', 'is_followed_artist']
    data = pd.read_csv(file, usecols=[0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], names = colNames, header=None)
    processed_data = {}
    user_ids = data['user_id'].unique()
    for user_id in user_ids:
        user_tracks = data[(data['user_id'] == user_id)]
        num_tracks = len(user_tracks.index)
        feature_sum = np.zeros(9)
        for idx in range(num_tracks):
            track = user_tracks.iloc[idx]
            weighted_features = np.array(get_feature_values(track)) * get_feature_weight(track)
            feature_sum = feature_sum + weighted_features
        feature_sum /= num_tracks
        processed_data[user_id] = feature_sum
    return processed_data, user_ids

def process_track_data(file):
    colNames = ['track_id', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence', 'tempo']
    data = pd.read_csv(file, usecols=[1, 4, 5, 6, 7, 8, 9, 10, 11, 12], names = colNames, header=None)
    processed_data = {}
    track_ids = data['track_id'].unique()
    for track_id in track_ids:
        track = data[(data['track_id'] == track_id)].iloc[0]
        features = np.array(get_feature_values(track))
        processed_data[track_id] = features
    return processed_data, track_ids