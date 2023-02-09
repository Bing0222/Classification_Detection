# @Time    : 2023/2/7
# @Author  : Bing

import os
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision

PATH = "E:/data/450birds/"
weightpath = "E:/data/brids/resnet50-0676ba61.pth"
savepath = 'E:/data/450birds/weight/'
batch_size = 32
epochs = 10

devices = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dir = os.path.join(PATH, 'train')
val_dir = os.path.join(PATH, 'valid')
test_dir = os.path.join(PATH, 'test')

df = pd.read_csv(os.path.join(PATH, 'birds.csv'))

train = df[df['data set'] == 'train'].reset_index(drop=True)
test = df[df['data set'] == 'test'].reset_index(drop=True)

train['filepaths'] = PATH + train['filepaths']
test['filepaths'] = PATH + test['filepaths']

num_classes = len(train.labels.unique())


def func(dir):
    directory = os.listdir(dir)
    for folder in directory:
        filename = os.path.join(folder)
        images = os.listdir(filename)

        print(f"The species is {folder} and contains {len(images)}")


# -------------------------------------------------- #
# （2）dataset
# -------------------------------------------------- #
transform_train = transforms.Compose([
    # 数据增强，随机裁剪224*224大小
    transforms.RandomResizedCrop(224),
    # 数据增强，随机水平翻转
    transforms.RandomHorizontalFlip(),
    # 数据变成tensor类型，像素值归一化，调整维度[h,w,c]==>[c,h,w]
    transforms.ToTensor(),
    # 对每个通道的像素进行标准化，给出每个通道的均值和方差
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# 验证集的数据预处理
transform_val = transforms.Compose([
    # 将输入图像大小调整为224*224
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

train_dataset = datasets.ImageFolder(root=PATH + 'train', transform=transform_train)
val_dataset = datasets.ImageFolder(root=PATH + 'test', transform=transform_val)

class_dict = train_dataset.class_to_idx
class_name = list(class_dict.keys())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

train_img, train_label = iter(train_loader).next()  # torch.Size([64, 3, 224, 224]) torch.Size([64])


def show_img(img):
    # img = train_img[:9] # [9, 3, 224, 224]
    img = img / 2 + 0.5
    img = img.numpy()
    class_label = train_label.numpy()
    img = np.transpose(img, [0, 2, 3, 1])
    plt.figure()
    for i in range(img.shape[0]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(img[i])
        plt.axis('off')
        plt.title(class_name[class_label[i]])
    plt.tight_layout()
    plt.show()


# show_img(train_img[:9])

# -------------------------------------------------- #
# （3）load model
# -------------------------------------------------- #


pretrained_dict = torch.load(weightpath, map_location=devices)
net = torchvision.models.resnet50(num_classes=num_classes)
model_dict = net.state_dict()
# print(net)

in_features = net.fc.in_features
net.fc = nn.Linear(in_features=in_features, out_features=num_classes)
# print(net)
net.to(devices)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.2)

best_acc = 0

torch.manual_seed(420)

for epoch in range(epochs):
    print('-' * 30, '\n', 'epoch:', epoch)
    net.train()
    running_loss = 0.0
    for idx, data in enumerate(train_loader):
        img, labels = data
        img = img.to(devices)
        labels = labels.to(devices)
        optimizer.zero_grad()
        output = net(img)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(f'step:{idx} loss:{loss}')

    net.eval()
    acc = 0.0

    with torch.no_grad():
        for data_set in val_loader:
            test_imgs, test_labels = data_set
            test_imgs = test_imgs.to(devices)
            test_labels = test_labels.to(devices)

            outputs = net(test_imgs)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == test_labels).sum().item()

        acc_test = acc / len(val_dataset)

        print(f'total_train_loss:{running_loss / idx},total_test_acc:{acc_test}')

        if acc_test > best_acc:
            best_acc = acc_test
            savename = savepath + 'resnet50.pth'
            torch.save(net.state_dict(), savename)
