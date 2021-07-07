import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from torch.utils.data import ConcatDataset, DataLoader, Subset,Dataset
from torchvision.datasets import DatasetFolder
import matplotlib
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
class my_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices,labels):
        self.dataset = dataset
        self.indices = indices
        labels_hold = torch.ones(len(dataset)).type(torch.long) *300 #( some number not present in the #labels just to make sure
        labels_hold[self.indices] = labels
        self.labels = labels_hold
    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]
        label = self.labels[self.indices[idx]]
        return (image, label)

    def __len__(self):
        return len(self.indices)

train_tfm = transforms.Compose([

    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),#随机水平翻转
    transforms.RandomRotation(15),
    # transforms.RandomGrayscale(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

#-----------------------------------------


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # 输入[3, 128, 128]
        #卷积层
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            #-------------
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

        )
        # 线性全连接网络
        self.fc_layers = nn.Sequential(

            nn.Linear(512 * 4 * 4, 1024),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)

        )

    def forward(self, x):
        # 输入 [batch_size, 3, 128, 128]
        # 输出 [batch_size, 11]

        x = self.cnn_layers(x)
        x = x.flatten(1)

        x = self.fc_layers(x)
        return x
def get_pseudo_labels(dataset, model, threshold=0.07):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=False, num_workers=0)
    x=[]
    y=[]
    model.eval()
    softmax = nn.Softmax()
    counter = 0
    for batch in tqdm(data_loader):
        img, _ = batch

        with torch.no_grad():
            logits = model(img.to(device))

        probs = softmax(logits)
        dataset.targets = torch.tensor(dataset.targets)

        for p in probs:
            if torch.max(p) >= threshold:
                if not (counter in x):
                    x.append(counter)
                my_subset(dataset,counter,torch.argmax(p))

            counter += 1
    model.train()
    new = Subset(dataset,x)
    print(len(new))
    return new
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set = DatasetFolder("food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg",
                              transform=train_tfm)
    valid_set = DatasetFolder("food-11/validation", loader=lambda x: Image.open(x), extensions="jpg",
                              transform=test_tfm)
    unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg",
                                  transform=train_tfm)
    test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
    batch_size =64

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Initialize a model, and put it on the device specified.
    model = Classifier().to(device)
    model.device = device

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    n_epochs = 100

    do_semi = True
    Loss_list = []
    Accuracy_list = []
    val_Loss_list = []
    val_Accuracy_list = []
    for epoch in range(n_epochs):
        if do_semi and epoch>=5:

            pseudo_set = get_pseudo_labels(unlabeled_set, model,0.85)
            concat_dataset = ConcatDataset([train_set, pseudo_set])
            train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

        # ---------- Training ----------
        model.train()
        train_loss = []
        train_accs = []

        for batch in tqdm(train_loader):

            imgs, labels = batch

            logits = model(imgs.to(device))

            loss = criterion(logits, labels.to(device))

            optimizer.zero_grad()

            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            optimizer.step()

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)


        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        Loss_list.append(train_loss)
        Accuracy_list.append(100 * train_acc)
        # ---------- Validation ----------
 
        model.eval()

        valid_loss = []
        valid_accs = []


        for batch in tqdm(valid_loader, position=0):

          with torch.no_grad():
              logits = model(imgs.to(device))


              loss = criterion(logits, labels.to(device))

              acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

              valid_loss.append(loss.item())
              valid_accs.append(acc)

        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)
        val_Loss_list.append(valid_loss)
        val_Accuracy_list.append(100*valid_acc)
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


    torch.save(model.state_dict(), 'CNN.model')
    model.eval()
    predictions = []
    test_accs = []

    for batch in tqdm(test_loader,position=0):
        imgs, labels = batch
 
        with torch.no_grad():
            logits = model(imgs.to(device))

        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    # print(test_acc)
    #绘图
    x1 = range(0, n_epochs)
    x2 = range(0, n_epochs)
    x3 = range(0, n_epochs)
    x4 = range(0, n_epochs)
    y1 = Accuracy_list
    y2 = Loss_list
    y4 =val_Loss_list
    y3 =val_Accuracy_list
    plt.subplot(2, 2, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Train accuracy vs. epoches')
    plt.ylabel('Train accuracy')
    plt.subplot(2, 2, 2)
    plt.plot(x2, y2, '.-')
    plt.title('Train loss vs. epoches')
    plt.xlabel('Train loss vs. epoches')
    plt.ylabel('Train loss')

    plt.subplot(2, 2, 3)
    plt.plot(x3, y3, color="red")
    plt.title('Validation accuracy vs. epoches')
    plt.ylabel('Validation accuracy')

    plt.subplot(2, 2, 4)
    plt.plot(x4, y4, color="red")
    plt.title('Validation loss vs. epoches')
    plt.ylabel('Validation loss')
    plt.show()

    with open("submission.csv", "w") as f:

        f.write("Id,Category\n")

        for i, pred in enumerate(predictions):

            f.write(f"{int(i)},{int(pred)}\n")