import pandas as pd
import torch.utils.data as utils
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
import os
import numpy as np


def data_loader(batch):
    """
    :param batch: Size of batches to load data in
    :return: dataloaders
    """
    # get df of features and labels
    data_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data')) + '\\features\\features_train_salary.csv.gz'
    df = pd.read_csv(data_path)

    # split into features and labels
    labels = df.loc[:, "%ofcap"]
    features = df.drop(["cap", "%ofcap", "Unnamed: 0", 'team_year', 'name_year'], axis=1)

    # get size no of labels and features and examples
    label_size = 1
    feature_size = features.shape[1]

    # put into tensors
    features_tensor = torch.tensor(features.values)
    labels_tensor = torch.tensor(labels.values)

    # put features, labels into dataset
    dataset = utils.TensorDataset(features_tensor, labels_tensor)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)

    return train_loader, label_size, feature_size


class Network(nn.Module):
    def __init__(self, input, hidden1, hidden2, output, dropout):
        super().__init__()

        # Defining the layers,
        self.fc1 = nn.Linear(input, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # forward pass to return output

        x = self.fc1(x)
        x = self.dropout(F.relu(x))
        x = self.fc2(x)
        x = self.dropout(F.relu(x))
        x = self.fc3(x)

        return x


def train(batch, epochs, hidden1, hidden2, lrate, dropout):
    """
    :param batch: Batch size
    :param epochs: Number of epochs to run for
    :param hidden1: Number of units in first hidden layer
    :param hidden2: Number of units in second layer
    :param lrate: Learning rate
    :param dropout: Probability of dropout
    :return: Saves trained model
    """

    # set random seeds so results are deterministic
    torch.manual_seed(0)
    np.random.seed(0)

    # get dataloaders and input, output size
    train_loader, output, input = data_loader(batch)

    # define model
    model = Network(input, hidden1, hidden2, output, dropout)

    # define loss function
    criterion = nn.L1Loss()

    # define optimiser
    optimiser = optim.Adam(model.parameters(), lr=lrate)

    # iterate
    for e in range(epochs):

        # make a forwards pass upon each batch from the training set to update weights
        for features, labels in train_loader:
            # set to gradient changing mode
            model.train()

            # reset gradients to zero
            optimiser.zero_grad()

            # forward pass to get prediction using features
            prediction = model(features.float())

            # reshape labels to match prediction
            labels = labels.view(-1, 1)

            # find loss between labels and predictions
            loss = criterion(prediction, labels.float())

            # find gradient
            loss.backward()

            # update weights using gradients
            optimiser.step()

    data_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data')) + "\\" + "\\"
    torch.save(model, data_path + 'model_salary.pth')

if __name__ == '__main__':
    train(32, 900, 64, 8, 0.00003, 0.3)
