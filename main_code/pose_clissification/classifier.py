#author: wmingzhu
#date: 2024/07/02
import torch
from torch.utils.data import Dataset,DataLoader

class PoseDataset(Dataset):
    def __init__(self,data_path=None):
        with open(data_path,"r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        self.data_source = lines

    def __getitem__(self, item):
        line = self.data_source[item].split(",")
        return line[0],line[1].strip()

    def __len__(self):
        return len(self.data_source)

class PoseClassifier:
    pass

p = PoseDataset("./train_data/train_indexed.txt")
print(p.data_source)

