import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import nn



class CTDataset(Dataset):
  def __init__(self, filepath):
    self.x, self.y = torch.load(filepath)
    self.x = self.x / 255 #Divide by 255 to normalize pixel values to the range of [0, 1]
    self.y = F.one_hot(self.y, num_classes=10).to(float)
  def __len__(self):
    return self.x.shape[0]  #returns num of images
  def __getitem__(self, ix):
    return self.x[ix], self.y[ix]
  
class MyNeuralNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.Matrix1 = nn.Linear(28**2, 100)  # nn.Linear makes a link of nodes with param 1 being input size (our image sizes)
    self.Matrix2 = nn.Linear(100, 50)         # and the second param being output size
    self.Matrix3 = nn.Linear(50, 10)
    self.R = nn.ReLU()    # ReLU is a function that introduces non-linearity, since a forward pass is just a linear transformation
  def forward(self, x):
    x = x.view(-1, 28**2)   #Get x into correct format for the pass
    x = self.R(self.Matrix1(x))   # Introduce non-linearity at each layer and pass through all columns of neurons
    x = self.R(self.Matrix2(x))
    x = self.Matrix3(x)
    return x.squeeze()  #Remove single dimensions
