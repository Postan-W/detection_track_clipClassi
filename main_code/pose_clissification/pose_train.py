#author: wmingzhu
#date: 2024/07/02
import torch
import visdom
from torch.utils.data import DataLoader
from classifier import PoseClassifier,PoseDataset
device = torch.device("cuda")

batch_size = 32
lr = 0.01
epochs = 20
workers = 4
torch.manual_seed(1234)#随机种子一定时，每次运行程序随机生成的数值都是一样的，包括初始化权重

train_dataset = PoseDataset("./train_data/train_indexed.txt",mode="train")
val_dataset = PoseDataset("./train_data/train_indexed.txt",mode="val")
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,drop_last=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
print(len(train_dataset),len(val_dataset))
#评估
def evaluate(model,data):
    correct = 0
    total = len(data.dataset)

    for x,y in data:
        x,y = x.to(device),y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred,y).sum().float().item()

    return correct/total

if __name__ == "__main__":
    vis = visdom.Visdom()
    model = PoseClassifier(input_dim=28,num_classes=7).to(device)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = torch.nn.CrossEntropyLoss()  # 这个对象本身包含了softmax操作。并且使用时传入的标签y是一个数值，one-hot也是这个对象给做了
    # 记录最佳模型
    best_epoch, best_accuracy,best_epoch_loss = 0,0,float('inf')

    global_step = 0
    for epoch in range(epochs):
        current_epoch_loss = 0 #用来记录当前epoch的总loss
        for step, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_function(logits, y)
            current_epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            vis.line([loss.item()], [global_step], win='train_step_loss', update='append')
            global_step += 1

        vis.line([current_epoch_loss], [epoch], win='train_epoch_loss', update='append')
        if current_epoch_loss < best_epoch_loss:
            best_epoch_loss = current_epoch_loss
            print("当前epoch:{}的loss最小，值为:{}".format(epoch,best_epoch_loss))
            torch.save(model, "./models/best_epoch_in_train.pt")  # 验证集上得到的best模型未必就是最好的，这里保存训练集上loss最小的模型

        if epoch % 2 == 0:
            accuracy = evaluate(model, val_dataloader)
            if accuracy > best_accuracy:
                best_epoch = epoch
                best_accuracy = accuracy
                # torch.save(model.state_dict(), 'best_in_val.pt')#model.load_state_dict(torch.load("best_in_val.pt"))模型定义找不到的话会报错
                torch.save(model, './models/best_epoch_in_val.pt')#模型较小，直接保存完整模型，加载较方便，torch.load("best_in_val.pt")
            vis.line([accuracy], [global_step], win='val_acc', update="append")
            print("epoch:{},accuracy:{}".format(epoch, accuracy))
            print("best_epoch:{},best_accuracy:{}".format(best_epoch, best_accuracy))


