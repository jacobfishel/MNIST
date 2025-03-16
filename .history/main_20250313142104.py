import torch
import model.pth

f = model.pth.MyNeuralNet()

f.load_state_dict(torch.load("model.pth"))
print(f.state_dict)