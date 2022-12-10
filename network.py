import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import tqdm

class SpotifyNet(nn.Module):
    def __init__(self, user_size, track_size, feature_size):
        super(SpotifyNet, self).__init__()

        self.num_users = user_size
        self.num_tracks = track_size
        self.feature_size = feature_size

        # create embedding for both inputs
        self.user_embed = nn.Embedding(self.num_users, self.feature_size)
        self.track_embed = nn.Embedding(self.num_tracks, self.feature_size)

        # define layers
        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)
    
    def forward(self, users, tracks):
        # combine embedded layer results
        # user embedding = dot product of users and user embedding map (process_user_data)
        # track embedding = dot product of track and track embedding map (process_track_data)
        
        user_embed = self.user_embed(users)
        track_embed = self.track_embed(tracks)

        x = torch.cat([user_embed, track_embed], dim=-1)

        # go through layers
        x = self.fc1(x)
        x = f.relu(x)
        x = self.fc2(x)
        x = f.relu(x)

        # return prediction
        out = self.output(x)
        prediction = f.sigmoid(out)
        return prediction

    def inference(self, batch):
        users, tracks = batch
        return self.forward(users, tracks) # gets and returns prediction

    def loss(self, prediction, interactions):
        return nn.BCELoss()(prediction, interactions.view(-1, 1).float())




def train(model, optimizer, train_loader, epoch, log_interval):
    model.train()
    losses = []
    for batch_idx, (users, tracks, interacted) in enumerate(tqdm.tqdm(train_loader)):
        # get prediction
        optimizer.zero_grad()
        batch = (users, tracks)
        output = model(batch)
        pred = output.max(-1)[1]

        # calculate and print losses
        loss = model.loss(output, pred, interacted)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return np.mean(losses)

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (users, tracks, interacted) in enumerate(test_loader):
            # get prediction
            batch = (users, tracks)
            output = model(batch)
            pred = output.max(-1)[1]

            # calculate loss
            test_loss += model.loss(output, interacted, reduction='mean').item()
            correct_mask = pred.eq(interacted.view_as(pred))
            num_correct = correct_mask.sum().item()
            correct += num_correct

    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / (len(test_loader.dataset) * test_loader.dataset.sequence_length)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset) * test_loader.dataset.sequence_length,
        100. * correct / (len(test_loader.dataset) * test_loader.dataset.sequence_length)))
    return test_loss, test_accuracy