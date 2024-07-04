import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# FeedForward Neural Network class
class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, hidden_size6, hidden_size7, hidden_size8, negative_slope=0.01):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        self.fc5 = nn.Linear(hidden_size4, hidden_size5)
        self.fc6 = nn.Linear(hidden_size5, hidden_size6)
        self.fc7 = nn.Linear(hidden_size6, hidden_size7)
        self.fc8 = nn.Linear(hidden_size7, hidden_size8)
        self.fc9 = nn.Linear(hidden_size8, 1)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))  
        x = self.relu(self.fc2(x))  
        x = self.leaky_relu(self.fc3(x)) 
        x = self.relu(self.fc4(x))  
        x = self.leaky_relu(self.fc5(x))  
        x = self.relu(self.fc6(x))
        x = self.leaky_relu(self.fc7(x))  
        x = self.relu(self.fc8(x))  
        output = self.fc9(x)
        return output


# Function to evaluate model performance on validation and test set
def test_model(model, data_loader, device):
    model.eval()
    y_pred = []
    y_test = []

    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            y_pred.append(output.cpu().numpy())
            y_test.append(targets.cpu().numpy())

    y_test = np.concatenate(y_test).squeeze()
    y_pred = np.concatenate(y_pred).squeeze()

    return y_test, y_pred

# Custom Dataset class
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.values).view(-1, 1)
        self.num_features = X.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx,:], self.y[idx]