import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import SGD
import os
import tarfile
from torch.utils.data import Dataset, DataLoader