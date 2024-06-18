import torch
version = f"https://data.pyg.org/whl/torch-{torch.__version__}.html"
print("TORCH_VERSION:",version) # https://data.pyg.org/whl/torch-2.3.1+cu121.html

import torch_geometric

from torch_geometric.datasets import ShapeNet

dataset = ShapeNet(root=".", categories=["Table", "Chair", "Guitar", "Motorbike"]).shuffle()[:5000]
print("Number of Samples: ", len(dataset))
print("Sample: ", dataset[0])

import plotly.express as px
import random
def plot_3d_shape(pos):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    fig = px.scatter_3d(x=x, y=y, z=z, opacity=0.3)
    fig.show()

sample_idx = random.choice(range(5000))
plot_3d_shape(dataset[sample_idx].pos)
print(dataset[sample_idx].category)

cat_dict = {key: 0 for key in dataset.categories}
print(cat_dict)
for datapoint in dataset: cat_dict[dataset.categories[datapoint.category.int()]]+=1
cat_dict

from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

augmentation = T.Compose([T.RandomJitter(0.01), T.RandomFlip(1), T.RandomShear(0.3)])

sample_idx = random.choice(range(5000))
plot_3d_shape(dataset[sample_idx].pos)
plot_3d_shape(augmentation(dataset[sample_idx]).pos)
category_key = dataset[sample_idx].category.int().item()
print(["Table", "Chair", "Guitar", "Motorbike"][category_key])

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool

class CL_model(torch.nn.Module):
  def __init__(self,k=20,aggr="max"):
    super().__init__()
    self.conv1 = DynamicEdgeConv(MLP([2*3,64,64]),k,aggr)#特征提取时 相对应每一条边需要提取的特征数量（x，y，z） 大小 节点的邻居 聚合操作
    self.conv2 = DynamicEdgeConv(MLP([2*64,128]),k,aggr)

    self.lin1 = Linear(128 + 64, 128)

    self.mlp = MLP([128, 256, 32], norm=None)

  def forward(self, data, train=True):
        if train:
            # Get 2 augmentations of the batch
            augm_1 = augmentation(data)
            augm_2 = augmentation(data)

            # Extract properties
            pos_1, batch_1 = augm_1.pos, augm_1.batch
            pos_2, batch_2 = augm_2.pos, augm_2.batch

            # Get representations for first augmented view
            x1 = self.conv1(pos_1, batch_1)
            x2 = self.conv2(x1, batch_1)
            h_points_1 = self.lin1(torch.cat([x1, x2], dim=1))

            # Get representations for second augmented view
            x1 = self.conv1(pos_2, batch_2)
            x2 = self.conv2(x1, batch_2)
            h_points_2 = self.lin1(torch.cat([x1, x2], dim=1))
            
            # Global representation
            h_1 = global_max_pool(h_points_1, batch_1)
            h_2 = global_max_pool(h_points_2, batch_2)
        else:
            x1 = self.conv1(data.pos, data.batch)
            x2 = self.conv2(x1, data.batch)
            h_points = self.lin1(torch.cat([x1, x2], dim=1))
            return global_max_pool(h_points, data.batch)

        # Transformation for loss function
        compact_h_1 = self.mlp(h_1)
        compact_h_2 = self.mlp(h_2)
        return h_1, h_2, compact_h_1, compact_h_2

    #当用positive labels训练的时候 loss变小 意味着同时在区分negative labels时的能力增强
    
from pytorch_metric_learning.losses import NTXentLoss
loss_func = NTXentLoss(temperature=0.10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CL_model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

import tqdm

def train():
    print("Training")
    model.train()
    total_loss = 0
    for _, data in enumerate(tqdm.tqdm(data_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        h_1, h_2, compact_h_1, compact_h_2 = model(data)
        embeddings = torch.cat((compact_h_1, compact_h_2))
        
        # The same index corresponds to a positive pair
        indices = torch.arange(0, compact_h_1.size(0), device=compact_h_2.device)
        labels = torch.cat((indices, indices))
        loss = loss_func(embeddings, labels)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(dataset)

for epoch in range(1, 2):
    loss = train()
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}')
    scheduler.step()

# torch.save(model.state_dict(), "./model")

from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sample = next(iter(data_loader))

# model = model(*args, **kwargs)
# model.load_state_dict(torch.load("./model/"))
# model.eval()

h = model(sample.to(device), train=False)
h = h.cpu().detach()
labels = sample.category.cpu().detach().numpy()

h_embedded = TSNE(n_components=2, learning_rate='auto',
                   init='random').fit_transform(h.numpy())

ax = sns.scatterplot(x=h_embedded[:,0], y=h_embedded[:,1], hue=labels, 
                    alpha=0.5, palette="tab10")

annotations = list(range(len(h_embedded[:,0])))

def label_points(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(int(point['val'])))

label_points(pd.Series(h_embedded[:,0]), 
            pd.Series(h_embedded[:,1]), 
            pd.Series(annotations), 
            plt.gca()) 

plt.show()