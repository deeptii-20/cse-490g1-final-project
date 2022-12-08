import process_data as pd

print("loading data...")
dataset = pd.loadData('data/spotify_dataset.csv')
train, test = pd.getTrainTestData(dataset)
train_data = pd.SpotifyDataset(train)
test_data = pd.SpotifyDataset(test)
print("done")
print()

print("making model...")
# batch = 128
# iters = 1000
# rate = .01
# momentum = .9
# decay = .005
# m = conv_net() # TODO
# print("done")
# print()

# print("training...")
# train_image_classifier(m, train, batch, iters, rate, momentum, decay) # TODO
# print("done")
# print()

# print("evaluating model...")
# print("training accuracy: %f", accuracy_net(m, train)) # TODO
# print("test accuracy:     %f", accuracy_net(m, test)) # TODO
# print()