import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# stores user, track, and interaction data from passed in dataSet
class SpotifyDataset(Dataset):
    def __init__(self, dataSet):
        self.users = dataSet['user_id']
        self.tracks = dataSet['track_id']
        self.interacted = dataSet['interacted']

    def __len__(self):
        return len(self.users)
  
    def __getitem__(self, idx):
        return self.users[idx], self.tracks[idx], self.interacted[idx]

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
    return processed_data

def process_track_data(file):
    colNames = ['track_id', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence', 'tempo']
    data = pd.read_csv(file, usecols=[1, 4, 5, 6, 7, 8, 9, 10, 11, 12], names = colNames, header=None)
    processed_data = {}
    track_ids = data['track_id'].unique()
    for track_id in track_ids:
        track = data[(data['track_id'] == track_id)].iloc[0]
        features = np.array(get_feature_values(track))
        processed_data[track_id] = features
    return processed_data

# TODO: remove
def loadData(dataFile):
    # get user_id and track_id
    colNames = ['user_id', 'track_id']
    dataSet = pd.read_csv(dataFile, usecols=[0, 1], names = colNames, header=None)

    # add interacted column for tracks already listened to
    dataSet['interacted'] = [1 for i in range(len(dataSet['user_id']))]

    # add 5 tracks not interacted with per tracks interacted with for each user
    for user_id in dataSet['user_id']:
        user_tracks = dataSet[(dataSet['user_id'] == user_id)]['track_id'].to_numpy().tolist()
        non_interacted_tracks = [track for track in dataSet['track_id'] if track not in user_tracks]
        for i in range(5):
            # if there are no more tracks to add, move on to the next user
            if len(non_interacted_tracks) <= 0:
                continue
            # otherwise, randomly select a track and remove from options
            new_track_id = np.random.choice(non_interacted_tracks)
            non_interacted_tracks.remove(new_track_id)
            # add user, new track to dataset mapping
            dataSet.loc[len(dataSet.index)] = [user_id, new_track_id, -1]
    return dataSet

# Creates training and testing dataframes. Training data is 80% of randomly selected tracks per user and testing
# data is the remaining 20% of the tracks.
# @param dataframe containing user_id, track_id, and interacted
# @returns [train dataframe, test dataframe]
def getTrainTestData(data):
    all_user_ids = data['user_id'].unique()
    train = []
    test = []

    for user_id in all_user_ids:
        # get all rows for a user
        user_rows = data.loc[data['user_id'] == user_id]

        # randomly split data into training and testing data (appending list faster than appending dataframes)
        curr_train = user_rows.sample(frac=0.8)
        
        for row in curr_train.to_numpy().tolist():
            train.append(row)

        curr_test = user_rows.drop(curr_train.index)
        for row in curr_test.to_numpy().tolist():
            test.append(row)

    colNames = ['user_id', 'track_id', 'interacted']
    return pd.DataFrame(train, columns=colNames), pd.DataFrame(test, columns=colNames)