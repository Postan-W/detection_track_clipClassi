import torch
t = torch.Tensor([[-1,2,3,4,5,-6,7,8,9]])
t = [round(i,2) for i in torch.nn.functional.softmax(t,dim=1).tolist()[0]]
t2 = torch.Tensor([[-1,2,3,4,5,-6,7,8,9]])
t2 = t2 / 5
t2 = [round(i,2) for i in torch.nn.functional.softmax(t2,dim=1).tolist()[0]]
print(t)
print(t2)
t2 = sorted(t2)
print(t2)

