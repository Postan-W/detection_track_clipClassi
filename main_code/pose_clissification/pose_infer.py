import torch
device = torch.device("cuda")
model = None
model.eval()#在评估模式下，所有特定于训练的层(dropout和batchnorm等)将被设置为不活动

with torch.no_grad():
    pass