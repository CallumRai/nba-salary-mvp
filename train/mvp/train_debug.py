import pandas as pd
import torch.utils.data as utils
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
import time
import matplotlib.pyplot as plt
import datetime
import os
import numpy as np


def data_loader(batch, test_prop):
    """
    :param batch: Size of batches to load data in
    :param test_prop: Proportion of data to be used as validation
    :return: Test and train dataloaders
    """
    # get df of features and labels
    data_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data')) + \
                '\\features\\features_train_mvp.csv.gz'
    df = pd.read_csv(data_path)

    # split into features and labels
    labels = df.loc[:, "Share"]
    features = df.drop(["Share", "Unnamed: 0", 'team_year', 'name_year'], axis=1)

    # get size no of labels and features and examples
    label_size = 1
    feature_size = features.shape[1]

    # put into tensors
    features_tensor = torch.tensor(features.values)
    labels_tensor = torch.tensor(labels.values)

    # put features, labels into dataset
    dataset = utils.TensorDataset(features_tensor, labels_tensor)

    # get size of test, train datests, dump, set
    test_size = int(len(df) * test_prop)
    train_size = len(df) - test_size

    # split datatest
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    # put sets into loaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch, shuffle=True)

    return train_loader, test_loader, label_size, feature_size


class Network(nn.Module):
    def __init__(self, input, hidden1, hidden2, output, dropout):
        super().__init__()

        # Defining the layers,
        self.fc1 = nn.Linear(input, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # forward pass to return output (sigmoid at end as value should be between 0,1)

        x = self.fc1(x)
        x = self.dropout(F.relu(x))
        x = self.fc2(x)
        x = self.dropout(F.relu(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)

        return x


def train_debug(batch, test_prop, epochs, hidden1, hidden2, lrate, dropout):
    """
        :param test_prop: Proportion of data to be used as validation
        :param batch: Batch size
        :param epochs: Number of epochs to run for
        :param hidden1: Number of units in first hidden layer
        :param hidden2: Number of units in second layer
        :param lrate: Learning rate
        :param dropout: Probability of dropout
        :return: Graph showing validation/test error and min validation error and epoch of occurance
        """

    # set random seeds so results are deterministic
    torch.manual_seed(0)
    np.random.seed(0)

    # get dataloaders and input, output size
    train_loader, test_loader, output, input = data_loader(batch, test_prop)

    # define model
    model = Network(input, hidden1, hidden2, output, dropout)

    # define loss function
    criterion = nn.BCELoss()

    # define optimiser
    optimiser = optim.Adam(model.parameters(), lr=lrate)

    # create empty loss lists
    train_loss = []
    test_loss = []

    # for displaying tiem left
    marker = 0
    time_running = 0

    # for displaying lowest loss
    min_loss = 1000000000000
    min_epoch = 0

    # iterate
    for e in range(epochs):
        start = time.time()

        # reset running losses
        train_running = 0
        test_running = 0

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

            # update running loss
            train_running += loss.item()

            # find gradient
            loss.backward()

            # update weights using gradients
            optimiser.step()

        # validate loss on test loader
        # turn of grads as we are not updating weights
        with torch.no_grad():
            model.eval()

            for features, labels in test_loader:
                # forward pass to get prediction using features
                prediction = model(features.float())

                # reshape labels to match prediction
                labels = labels.view(-1, 1)

                # calculate loss
                loss = criterion(prediction, labels.float())

                # update running loss
                test_running += loss.item()

        # append losses
        train_loss.append(train_running / len(train_loader))
        test_loss.append(test_running / len(test_loader))

        # update min loss
        if test_running / len(test_loader) < min_loss:
            min_loss = test_running / len(test_loader)
            min_epoch = e

        # time displaying
        end = time.time()

        if marker == 20:
            time_running += (end - start)
            time_estimate = (time_running / 20) * (epochs - e)
            time_estimate = str(datetime.timedelta(seconds=time_estimate))
            print(f"{time_estimate} - Remaining")
            marker = 0
            time_running = 0
        else:
            time_running += (end - start)
            marker += 1

    print(f"Min loss = {min_loss} at epoch {min_epoch}")
    print(f"Final training loss - {train_loss[-1]}")

    # plot test and train loss
    plt.plot(train_loss, label='Training loss')
    plt.plot(test_loss, label='Validation loss')
    plt.ylim(0, 0.1)
    plt.legend(frameon=False)
    plt.show()


if __name__ == '__main__':
    train_debug(32, 0.2, 500, 32, 8, 0.00003, 0.2)
