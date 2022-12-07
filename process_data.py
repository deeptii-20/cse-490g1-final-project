import pandas as pd
import numpy as np

# Processes data from the spotify csv file
# @param dataFile path to a csv file that contains information about a user's spotify playlist
# @returns a dataframe with columns user_id, track_id, and interacted. Interacted will be 1
# if the user has interacted with the track before (has listened to the track before, 
# has added the track to a playlist, or has the track as one of their top songs) and -1 if not. It has a
# 5:1 ratio of tracks that a user hasn't and has interacted with
def loadData(dataFile):
    # get user_id and track_id
    colNames = ['user_id', 'track_id']
    dataSet = pd.read_csv(dataFile, names=colnames, header=None)

    # create mapping of user_ids and track_ids to keep track of what the user has listened to
    track_mapping = {}
    for i in range len(dataSet['userId']):
        # get current track and user id
        curr_user = dataSet['user_id'][i]
        curr_track = dataSet['track_id'][i]

        # update list of tracks user has listened to
        user_tracks = []
        if (user_id in track_mapping.keys()):
            user_tracks = track_mapping.get(user_id)
        user_tracks.append(curr_track)

        # add track list to map
        track_mapping[curr_user] = user_tracks

    # add interacted column for tracks already listened to
    dataSet['interacted'] = [1 for i in range(len(dataSet['userId']))]

    # add 5 tracks not interacted with per tracks interacted with for a user
    all_tracks = dataSet['track_id'].unique()
    for user_id in dataSet['user_id']:
        for i in range(5):
            # randomly select a tracl
            new_track_id = np.random.choice(all_tracks)

            # keep randomly selecting a track until the user hasn't interacted with it
            user_tracks = track_mapping[user_id]
            while new_track_id in user_tracks:
                new_track_id = np.random.choice(all_tracks)

            # add user, new track to dataset mapping
            dataSet.loc[len(dataSet.index)] = [user_id, new_track_id, 0]

    return dataSet

# Creates training and testing dataframes. Training data is 80% of randomly selected tracks per user and testing
# data is the remaining 20% of the tracks.
# @param dataframe containing user_id, track_id, and interacted
# @returns [train dataframe, test dataframe]
def getTrainTestData(data):
    all_user_ids = data['user_id'].unique()
    train = []
    test = []

    for user_id in user_ids:
        # get all rows for a user
        user_rows = data.loc[data['user_id'] == user_id]

        # randomly split data into training and testing data (appending list faster than appending dataframes)
        curr_train = user_rows.sample(frac=0.8).to_numpy().tolist()
        for row in curr_train:
            train.append(row)

        curr_test = user_rows.drop(train.index).to_numpy().tolist()
        for row in curr_train:
            train.append(row)

    colNames = ['user_id', 'track_id', 'interacted']
    [pd.DataFrame(train, columns=colNames), pd.DataFrame(test, columns=colNames)]