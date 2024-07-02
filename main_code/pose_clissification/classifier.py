#author: wmingzhu
#date: 2024/07/02
import torch
from torch.utils.data import Dataset,DataLoader

class PoseDataset(Dataset):
    def __init__(self,data_path=None,mode="train"):
        with open(data_path,"r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            total = len(lines)
            #上面做下shuffle可能更合适，以免train或val的数据单一。这里未做，假定是前0.8和后0.2的数据都覆盖了各种类型
            if mode == "train":
                lines = lines[:int(0.8*len(lines))]
            elif mode == "val":
                lines = lines[int(0.8 * len(lines)):]

        self.data_source = lines

    def __getitem__(self, item):
        line = self.data_source[item].split(",")
        label = torch.tensor(int(line[0]),dtype=torch.float32)
        data = torch.tensor([float(i) for i in line[1:]],dtype=torch.float32)
        return data,label

    def __len__(self):
        return len(self.data_source)

class PoseClassifier(torch.nn.Module):
    def __init__(self, input_dim,num_classes):
        super(PoseClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim,256)
        self.activation = torch.nn.LeakyReLU(0.01)
        # self.drop = torch.nn.Dropout(0.25)#模型比较简单，暂时不用
        self.linear2 = torch.nn.Linear(256,128)
        self.output = torch.nn.Linear(128,num_classes)

    def forward(self, x):
        #注意：多分类损失函数nn.CrossEntropyLoss()内部做了softmax和标签的one-hot，所以不用我们显式做这两个东西
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        out = self.activation(out)
        out = self.output(out)
        return out


if __name__ == "__main__":
    train_dataset = PoseDataset("./train_data/train_indexed.txt",mode="train")
    val_dataset = PoseDataset("./train_data/train_indexed.txt",mode="val")
    print(len(train_dataset),len(val_dataset))
    dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers=4,drop_last=True)
    for i in dataloader:
        print(i)


