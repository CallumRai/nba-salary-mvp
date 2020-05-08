import pandas as pd
import torch.nn.functional as F
from torch import nn
import os
import torch


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
        x = torch.sigmoid(x)

        return x

def predicted_mvp():
    """
    :return: Dataframe of predicted mvp share 1990-2020
    """

    # load feature df
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + '\\'
    df = pd.read_csv(data_path + "features\\features_pred_mvp.csv.gz", encoding='utf-8-sig', compression='gzip')

    # load model
    model = torch.load(data_path + "model_mvp.pth")
    model.eval()

    # get prediciton of  %ofcap
    df["pred_mvp_share"] = df.apply(lambda row: model(torch.Tensor(row.drop(["Share", "Unnamed: 0", 'team_year', 'name_year'])
                                                                  .tolist())).item(), axis=1)

    return df

def predicted_mvp_2020():
    """
    :return: Dataframe of predicted mvp share 1990-2020
    """

    # load feature df
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + '\\'
    df = pd.read_csv(data_path + "features\\features_2020_mvp.csv.gz", encoding='utf-8-sig', compression='gzip')

    # load model
    model = torch.load(data_path + "model_mvp.pth")
    model.eval()

    # get prediciton of  %ofcap
    df["pred_mvp_share"] = df.apply(lambda row: model(torch.Tensor(row.drop(["Unnamed: 0", 'team_year', 'name_year'])
                                                                  .tolist())).item(), axis=1)

    return df


if __name__ == '__main__':
    # saves all csvs
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + '\\predictions\\'

    pred_mvp = predicted_mvp()
    pred_mvp.to_csv(data_path + "pred_mvp.csv.gz", encoding='utf-8-sig', compression='gzip')

    future_mvp = predicted_mvp_2020()
    future_mvp.to_csv(data_path + "2020_mvp.csv.gz", encoding='utf-8-sig', compression='gzip')