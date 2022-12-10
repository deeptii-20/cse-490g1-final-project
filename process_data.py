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

    def __len__(self): # returns the number of unique (user, track) pairings
        return len(self.users)
  
    def __getitem__(self, idx):
        return self.users[idx], self.tracks[idx], self.interacted[idx]

    def user_size(self):
        return len(self.users)

    def track_size(self):
        return len(self.tracks)

# Processes data from the spotify csv file
# @param dataFile path to a csv file that contains information about a user's spotify playlist
# @returns a dataframe with columns user_id, track_id, and interacted. Interacted will be 1
# if the user has interacted with the track before (has listened to the track before, 
# has added the track to a playlist, or has the track as one of their top songs) and -1 if not. It has a
# 5:1 ratio of tracks that a user hasn't and has interacted with
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