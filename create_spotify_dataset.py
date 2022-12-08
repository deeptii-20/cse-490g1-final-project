import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import time

CLIENT_ID = '512eb40fa7af4c8db0864090d056ccfb'  # '43987e690b184efa9ed09e56b78f10bd'
CLIENT_SECRET = '5afbc098885444848fcbcdf5efa94b3e'  # 'e1b4e8ce90e846368204b1c8c33d3410'
SCOPE = "user-library-read user-read-recently-played user-follow-read"
REDIRECT_URI = "http://localhost/"

def get_filtered_features(features_list):
    filtered_features = {}
    for feature_name in ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'valence', 'tempo']:
        filtered_features[feature_name] = features_list[0][feature_name]
    return filtered_features

def create_spotify_dataset():
    """
    run this code once for each user
    make sure to log out after each iteration
    """
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, scope=SCOPE, redirect_uri=REDIRECT_URI))
    
    # get users' preferences from playlists, saved tracks, recently played tracks, and followed artists -> save info in a dataframe
    spotify_df = pd.DataFrame(columns=["user_id", "track_id", "track_name", "artist", "features", "is_on_playlist", "is_saved", "is_recently_played", "is_followed_artist"])
    # get the user's name
    user_id = sp.current_user()["display_name"]
    
    # # go through the user's playlists
    for playlist in sp.current_user_playlists(limit=25)["items"]:
        for track in sp.playlist_tracks(playlist["id"])["items"]:
            track_id = track["track"]["id"]
            track_name = track["track"]["name"]
            artist = track["track"]["artists"][0]["name"]
            features = get_filtered_features(sp.audio_features(track_id))
            spotify_df.loc[(len(spotify_df.index))] = [user_id, track_id, track_name, artist, features, True, False, False, False]
    
    # # go through the user's saved tracks
    for offset_mult in range(6):
        for track in sp.current_user_saved_tracks(limit=50, offset=(offset_mult*50))["items"]:
            track_id = track["track"]["id"]
            if ((spotify_df["user_id"] == user_id) & (spotify_df["track_id"] == track_id)).any():
                spotify_df.loc[((spotify_df["user_id"] == user_id) & (spotify_df["track_id"] == track_id)), "is_saved"] = True
                continue
            track_name = track["track"]["name"]
            artist = track["track"]["artists"][0]["name"]
            features = get_filtered_features(sp.audio_features(track_id))
            spotify_df.loc[len(spotify_df.index)] = [user_id, track_id, track_name, artist, features, False, True, False, False]     
        
    # # go through the user's recently played tracks
    one_month_ago_timestamp = (int(time.time() * 1000)) - (3 * 2629800000)
    for track in sp.current_user_recently_played(limit=50, after=one_month_ago_timestamp)["items"]:
        track_id = track["track"]["id"]
        if ((spotify_df["user_id"] == user_id) & (spotify_df["track_id"] == track_id)).any():
            spotify_df.loc[((spotify_df["user_id"] == user_id) & (spotify_df["track_id"] == track_id)), "is_recently_played"] = True
            continue
        track_name = track["track"]["name"]
        artist = track["track"]["artists"][0]["name"]
        features = get_filtered_features(sp.audio_features(track_id))
        spotify_df.loc[len(spotify_df.index)] = [user_id, track_id, track_name, artist, features, False, False, True, False]    
    
    # go through the user's followed artists
    for artist in sp.current_user_followed_artists(limit=25)["artists"]["items"]:
        user_tracks_by_artist = spotify_df[((spotify_df['user_id'] == user_id) & (spotify_df['artist'] == artist["name"]))]['track_id'].to_numpy().tolist()
        for track in user_tracks_by_artist:
            spotify_df.loc[((spotify_df["user_id"] == user_id) & (spotify_df["track_id"] == track)), "is_followed_artist"] = True
        for track in sp.artist_top_tracks(artist["id"])["tracks"]:
            track_id = track["id"]
            if ((spotify_df["user_id"] == user_id) & (spotify_df["track_id"] == track_id)).any():
                spotify_df.loc[((spotify_df["user_id"] == user_id) & (spotify_df["track_id"] == track_id)), "is_followed_artist"] = True
                continue
            track_name = track["name"]
            artist = track["artists"][0]["name"]
            features = get_filtered_features(sp.audio_features(track_id))
            spotify_df.loc[len(spotify_df.index)] = [user_id, track_id, track_name, artist, features, False, False, False, True]
    
    # append the dataset to a csv
    spotify_df.to_csv('./data/spotify_dataset.csv', mode='a', index=False, header=False)

create_spotify_dataset()