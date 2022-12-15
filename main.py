import torch
import process_data as pd
import network as n
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

FILE_NAME = 'data/spotify_dataset.csv'

print("loading data...")
data, train, test = pd.get_train_test_data(FILE_NAME)
train_data = pd.SpotifyDataset(train)
test_data = pd.SpotifyDataset(test)
print("done")
print()

print("making model...")
TRAIN_BATCH_SIZE = 50
FEATURE_SIZE = 9
TEST_BATCH_SIZE = 50
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
PRINT_INTERVAL = 10
m = n.SpotifyNet(train_data.user_size(), train_data.track_size(), FEATURE_SIZE)
print("done")
print()

print("training...")
optimizer = optim.Adam(m.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
test_losses, test_accuracies, train_losses = [], [], []
try:
    for epoch in range(1, EPOCHS + 1):
        lr = LEARNING_RATE * np.power(0.25, (int(epoch / 6)))
        train_loss = n.train(m, optimizer, train_loader, epoch, PRINT_INTERVAL)
        test_loss, test_accuracy = n.test(m, test_loader)
        test_losses.append((epoch, test_loss))
        test_accuracies.append((epoch, test_accuracy))
        train_losses.append((epoch, train_loss))
except KeyboardInterrupt as ke:
        print('Interrupted')
except:
    import traceback
    traceback.print_exc()
finally:
    ep, val = zip(*train_losses)
    plt.plot(val)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Train Loss vs Epoch')
    plt.show()
    ep, val = zip(*test_losses)
    plt.plot(val)
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs Epoch')
    plt.show()
    ep, val = zip(*test_accuracies)
    plt.plot(val)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracies vs Epoch')
    plt.show()
print("done")
print()

print("generating song recommendations...")
# get track mapping
track_info = pd.track_mapping(FILE_NAME)

# get rows in all_data for the user
unique_users = list(set([pd.onehot_to_item(user_id) for user_id in data['user_id']]))
for user in unique_users:
    # get encoding
    user_encoding = pd.item_to_onehot(user)
    
    # get user rows from data
    user_indexes = []
    for i, row in data.iterrows():
        user_id = row['user_id']
        if user_id == user_encoding:
            user_indexes.append(i)
    user_rows = data.iloc[user_indexes]

    # convert each user_id and track_id to numpy arrays
    users_arr = np.asarray([np.array(user) for user in user_rows["user_id"]]) 
    tracks_arr = np.asarray([np.array(track) for track in user_rows["track_id"]])

    # get predicted scores
    predicted_scores = m(torch.LongTensor(users_arr), torch.LongTensor(tracks_arr))

    # store (track, score)
    track_scores = [(tracks_arr[idx], predicted_scores[idx].item()) for idx in range(len(predicted_scores))]

    # get top ten tracks with the highest score
    track_scores = sorted(track_scores, key = lambda x : x[1], reverse=True)[0:10]
    track_ids = [pd.onehot_to_item(track) for (track, score) in track_scores]
    
    # print out song recommendations
    print('Top 10 song recommendations for {}'.format(user))
    for i in range(len(track_scores)):
        track_name, track_artist = track_info.get(track_ids[i])
        print('{}. {} by {}'.format((i+1), track_name, track_artist))
    print()
print("done")
print()