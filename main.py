# modified from tryhw1

print("loading data...")
dataset = loadData('data/spotify_dataset.csv')
train, test = getTrainTestData(dataSet)
print("done")
print()

print("making model...")
batch = 128
iters = 1000
rate = .01
momentum = .9
decay = .005
m = conv_net() # TODO
print("done")
print()

print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print()

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))
print()