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

        return x

def predicted_salaries():
    """
    :return: Dataframe of predicted salaries using model from 1990 - 2020
    """

    # load feature df
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + '\\'
    df = pd.read_csv(data_path + "features\\features_pred_salary.csv.gz")

    # load model
    model = torch.load(data_path + "model_salary.pth")
    model.eval()

    # get prediciton of  %ofcap
    df["pred_%ofcap"] = df.apply(lambda row: model(torch.Tensor(row.drop(["cap", "Unnamed: 0", 'team_year', 'name_year'])
                                                                .tolist())).item(), axis=1)

    # get prediction of salary
    df["pred_salary"] = df.apply(lambda row: row["cap"]*row["pred_%ofcap"]/100, axis=1)

    return df

if __name__ == '__main__':
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + '\\predictions\\'

    pred_salary_df = predicted_salaries()
    pred_salary_df.to_csv(data_path + "pred_salary.csv.gz", encoding='utf-8-sig', compression='gzip')