import os
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
import torch
from torch import Tensor
import torch_geometric
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
import networkx as nx
from networkx.algorithms import community
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.nn import GCNConv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

dataset = Planetoid(root=data_dir, name='Cora')
data = dataset[0]

print(data.num_node_features)
print(dataset.num_classes)

train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask

# Access training data
x_train = data.x[train_mask]
y_train = data.y[train_mask]

# Access validation data
x_val = data.x[val_mask]
y_val = data.y[val_mask]

# Access test data
x_test = data.x[test_mask]
y_test = data.y[test_mask]
print(data.train_mask.sum().item())
# label_dict = {
#     0: "Theory",
#     1: "Reinforcement_Learning",
#     2: "Genetic_Algorithms",
#     3: "Neural_Networks",
#     4: "Probabilistic_Methods",
#     5: "Case_Based",
#     6: "Rule_Learning"}

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 67)
        self.conv2 = GCNConv(67, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=.951, training=self.training)  # Increased dropout

        
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# initialize the model
model = GCN(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)  # Increased weight decay and reduced learning rate
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Define the loss function (criterion)
criterion = torch.nn.CrossEntropyLoss()

# Early stopping parameters
patience = 10
best_val_loss = None
epochs_no_improve = 0

...

def train():
    model.train()
    optimizer.zero_grad() # Clear gradients.
    out = model(data) # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss.item()

def test(mask):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        loss = criterion(logits[mask], data.y[mask]).item()
    return acc, loss

data = data.to(device)  # Add this line

# Training loop
for epoch in range(500):
    train_loss = train()
    train_acc, _ = test(data.train_mask)
    val_acc, val_loss = test(data.val_mask)
    print(f'Epoch: {epoch+1:03d}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}')

    # Early stopping
    if best_val_loss is None:
        best_val_loss = val_loss
    elif val_loss > best_val_loss:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print("Early stopping!")
            break
    else:
        epochs_no_improve = 0
        best_val_loss = val_loss

    scheduler.step()

