# Spotify Recommendations
Deepti Ramani, Simran Malhi

---

## Abstract

In this project, we analyze the Spotify data of average users in order to predict compatibility scores for users and songs, then use these predicted scores to generate song recommendations for the users.

We use a Neural Collaborative Filtering model to predict compatibility scores, then output the highest-scoring tracks for each user as recommendations. However, our model was unsuccessful in its predictions: the predicted track scores converged at one value despite the wide variety in the actual scores, which resulted in the same ten songs getting recommended for each user, regardless of their listening habits and actual audio feature preferences. In the future, we plan to experiment with different model architectures and parameters to see if we can improve our model's ability to learn the user and track features.

---

## Problem

For our project, we wanted to be able to generate song recommendations for users based on their Spotify data, since it can be difficult to find new songs to listen to. 

We approached this problem using the following steps:

1. Collect Spotify data from various users and creating a dataset of users and the songs they have interacted with

2. Process the data to generate compatibility scores for each user/track combination

3. Train the model to learn the user's preferred audio features and predict a score for each user/track combination

4. Generate 10 recommended songs for each user by using the highest predicted compatbility scores

---

## Related Work

### Kaggle Recommendation Competition
We came across a [Kaggle competition](https://www.kaggle.com/c/movielens-100k) that involved generating movie recommendations using the MovieLens dataset. This idea inspired us to create a Spotify song deep learning recommendation system that uses Spotify's dataset.

### Our Chosen Dataset: The Spotify Dataset (Users, Tracks, and Scores)

The first dataset we used was a collection of Spotify data pulled from five user accounts using the [spotipy](https://spotipy.readthedocs.io/en/2.22.0/) library. 

For each user, we retrieved their unique user id and the tracks they had interacted with (added to a playlist, saved, recently listened to, by a followed artist). For each of these tracks, we stored its unique track id, the track name, artist, certain audio features, and the different ways in which the user had interacted with it.

We retrieved 9 audio features for each track:
1. acousticness
2. danceability
3. energy
4. instrumentalness
5. liveness
6. loudness
7. speechiness
8. valence
9. tempo

From the Spotify dataset, we created two additional datasets: one mapping each track to its audio features, and one mapping each user to their preferred audio features. In these datasets, the user and track ids were one-hot encoded and padded for consistent length.

Then, using the users' preferred audio features and the track audio features, we generated compatibility scores for each unique user/track combination, representing how much a user would like a track. This dataset stored 19360 scores (3872 songs for each of 5 users).

---

## Methodology


### Retrieving Spotify Data

Retrieving data from Spotify was handled by the `create_spotify_dataset.py` script. 

This involves three major steps:

1. Create a Spotify API client, prompting the current user to sign in

2. Retrieve the current user's user id, then iterate through playlists, saved tracks, recently played tracks, and followed artists to add tracks to the dataset

3. Write the dataset to a file, appending it to the end of any existing data (not overwriting)


### Processing Data

Processing Spotify data for training and testing our model was handled by the `process_data.py` script. 

This involves five major steps:
   
   1. `process_user_data` reads from the overall Spotify dataset and calculates the user's preferred feature values by taking a weighted average of the features of each track in the dataset
   
   2. `process_track_data` reads from the overall Spotify dataset and stores the feature values for each track 
   
   3. `item_to_onehot` and `onehot_to_item` are helper methods used to convert between user/track ids (stored as alphanumeric strings) and arrays containing 1s and 0s. Both user ids and track ids are padded to 30 characters during encoding, since this is the maximum allowed length for a Spotify user id.
   
   4. `get_scores` takes user feature preferences and track features and calculates a compatibility score for each user/track combination using a dot product. These scores get added to a dataset along with the one-hot encoded user ids and track ids
   
   5. `get_train_test_data` takes the dataset of scores and separates it into training and test data by randomly selecting 80% of the tracks for each user.


### Neural Network

We trained a Neural Collaborative Filtering model on the dataset in order to generate track scores for the users. This network and the train and test methods are defined in `network.py`.

The network consisted of 2 embedded layers (1 for users and 1 for tracks) to represent their traits in a lower dimensional space in order to learn the features of the users and tracks. We had 9 features for the embedded layers, since each user and track has 9 traits.

We then concatenate the user and track embeddings into one vector and pass it through a series of fully connected layers consisting of Linear layers with ReLU activation functions to map the embeddings to score predictions. The first linear layer takes in 18 features because the concatenated vector has 18 features (9 user features + 9 track features). The final linear layer had a output of 1 because we want one score prediction for each (user, track) pairing.

Finally, we then pass the predictions through a sigmoid function to ensure that they would be between 0 and 1, since the calculated scores were normalized between 0 and 1.

For our loss function, we chose to use the MSE loss function because our problem is a regression problem and we are trying to minimize the difference between the predicted and actual scores.

---

## Evaluation


### Experiments:

For our experiments, we experimented with changing our batch sizes, epochs, learning rate, and weight decay to improve our test accuracies. We gathered the user data for our experiments through running `create_spotify_dataset.py` for different Spotify users. We then ran our experiments by running `main.py`, which:

   1. Loaded and processed the training and testing data from `spotify_dataset.csv` through invoking `process_data.py`

   2. Initialized the parameters and created and trained the model through invoking `network.py`

   3. Generated the top ten song recommendations per user through sorting the predicted scores for each track (selecting the top ten songs with the highest predicted scores)

Our worst model had batch sizes of 512, 20 epochs, a learning rate of 0.1, and a weight decay of 0.0005, as this had a final epoch test accuracy of 26%.

Our best model had batch sizes of 50, 50 epochs, a learning rate of 0.001, and a weight decay of 0.0005, as this model had a final epoch test accuracy of 50%, which is what we stuck with.

We would have experimented more with different network architectures to find the most optimal one, however, as mentioned in the Results section, we weren't able to create a model that fulfilled all of our base evaluation criteria.


### Evaluation Criteria:

To evaluate the accuracy of the score predictions, we calculated the test accuracies for each epoch by subtracting the difference of each predicted and actual scores and counting the number of predicted values that had a difference in value less than 0.00000001. We determined that a final test accuracy greater than 30% would be acceptable.

To evaluate the accuracy of the recommender system as a whole, we manually reviewed each of the ten song recommendations per user. We determined that the evaluation would be a success if each user had reasonably different song recommendations that align with the songs that they listen to and contain at least one song that they have listened to before. 

---

## Results

Our results met the test accuracy criteria of being over 30%, as our final epoch had a test accuracy of 50%. However, this only meant that the predicted float values weren't very off from the actual values, which was easy to achieve since both the predicted and actual values were between 0 and 1. 

Unfortunately, our model didn't accurately recommend songs for each of the users. Our model generated the same predicted score values for every user, track pairing. Although each user had very distinct and diverse track lists, the model wasn't able to accurately learn the user and track features in order to generate many different score values within the same iteration. For example, the predicted scores for one user was ```[0.5373, 0.5373, 0.5373, 0.5373, 0.5185, 0.5373, 0.5373, 0.5373, 0.5373]``` when the actual scores were ```[0.6605, 0.4063, 0.3994, 0.4503, 0.4685, 0.4467, 0.6143, 0.3738, 0.4584]```. This lead to the same set of songs being recommended for every user. 

We have several possible ideas for why this model didn't work. However, because of time constraints, we weren't able to create a better model on time.

1. **We did not pick the right activation function.** 
    
    It is possible that the ReLU activation function caused the predicted values to converge too quickly, which would explain why almost all of the predicted scores were the same value. If we found a better activation function, it is possible that the model would have predicted a more diverse range of values for each track and have had more accurate song recommendations per user.

2. **We did not have enough hidden layers.** 
    
    It is possible that we didn't have enough layers for the model to find patterns between the input and output vectors and learn the features of the data with our given parameters (epoch, learning rate, weight decay, batch size).
    
3. **We did not have good model parameters, specifically the learning rate and weight decay.** 
    
    It is possible that our learning rate was still too large, which caused our model to keep overshooting. Our weight decay may have been too large, which caused our model to severely underfit the data and only create one score prediction at a time for each batch of users and tracks.

4. **We did not use the correct loss function.**
    We chose to use the MSE loss function we want to minimize the difference between the predicted and actual scores. However, it is possible that our loss function prevented our model from being able to properly fit the data and that there is a better loss function that we can use.
---

## Demo video

Demo video can be found [here](https://github.com/deeptii-20/cse-490g1-final-project/blob/main/cse490g1-final-project-video.mp4).

---

## Code

Code for this project can be found [here](https://github.com/deeptii-20/cse-490g1-final-project).