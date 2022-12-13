import torch
import process_data as pd
import network as n
import torch.optim as optim
import numpy as np

print("loading data...")
train, test = pd.getTrainTestData('data/spotify_dataset.csv')
train_data = pd.SpotifyDataset(train)
test_data = pd.SpotifyDataset(test)

print("making model...")
TRAIN_BATCH_SIZE = 256
FEATURE_SIZE = 9
TEST_BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0005
PRINT_INTERVAL = 10
m = n.SpotifyNet(train_data.user_size(), train_data.track_size(), FEATURE_SIZE)
print("done")
print()

print("training...")
optimizer = optim.Adam(m.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
train_losses, test_losses, test_accuracies = [], [], []
try:
    for epoch in range(1, EPOCHS + 1):
        lr = LEARNING_RATE * np.power(0.25, (int(epoch / 6)))
        train_loss = n.train(m, optimizer, train_loader, epoch, PRINT_INTERVAL)
        test_loss, test_accuracy = n.test(m, test_loader)
        # train_losses.append((epoch, train_loss))
        # test_losses.append((epoch, test_loss))
        # test_accuracies.append((epoch, test_accuracy))
except KeyboardInterrupt as ke:
        print('Interrupted')
except:
    import traceback
    traceback.print_exc()
print("done")
print()

print("evaluating model...")
# print("training accuracy: %f", accuracy_net(m, train_data)) # TODO: use hit 10
# print("test accuracy:     %f", accuracy_net(m, test_data)) # TODO: use hit 10

# ep, val = zip(*train_losses)
# pt_util.plot(ep, val, 'Train loss', 'Epoch', 'Error')
# ep, val = zip(*test_losses)
# pt_util.plot(ep, val, 'Test loss', 'Epoch', 'Error')
# ep, val = zip(*test_accuracies)
# pt_util.plot(ep, val, 'Test accuracy', 'Epoch', 'Error')
print()