# @Time    : 2023/2/3
# @Author  : Bing


import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ResNet import resnet50  # 从自定义的ResNet.py文件中导入resnet50这个函数
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------- #
# （0）参数设置
# -------------------------------------------------- #
batch_size = 32  # 每个step训练32张图片
epochs = 10  # 共训练10次

# -------------------------------------------------- #
# （1）文件配置
# -------------------------------------------------- #
# 数据集文件夹位置
filepath = r"E:/data/brids/"
# 权重文件位置
weightpath = r"E:\data\brids\resnet50-0676ba61.pth"
# 权重保存文件夹路径
savepath = r"E:/data/brids/weights/"

# 获取GPU设备
if torch.cuda.is_available():  # 如果有GPU就用，没有就用CPU
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# -------------------------------------------------- #
# （2）构造数据集
# -------------------------------------------------- #
# 训练集的数据预处理
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

# 读取训练集并预处理
train_dataset = datasets.ImageFolder(root=filepath + 'training_data',  # 训练集图片所在的文件夹
                                     transform=transform_train)  # 训练集的预处理方法

# 读取验证集并预处理
val_dataset = datasets.ImageFolder(root=filepath + 'predict_data',  # 验证集图片所在的文件夹
                                   transform=transform_val)  # 验证集的预处理方法

# 查看训练集和验证集的图片数量
train_num = len(train_dataset)
val_num = len(val_dataset)
print('train_num:', train_num, 'val_num:', val_num)  # 453, 112

# 查看图像类别及其对应的索引
class_dict = train_dataset.class_to_idx
print(class_dict)  # {'Bananaquit': 0, 'Black Skimmer': 1, 'Black Throated Bushtiti': 2, 'Cockatoo': 3}
# 将类别名称保存在列表中
class_names = list(class_dict.keys())

# 构造训练集
train_loader = DataLoader(dataset=train_dataset,  # 接收训练集
                          batch_size=batch_size,  # 训练时每个step处理32张图
                          shuffle=True,  # 打乱每个batch
                          num_workers=0)  # 加载数据时的线程数量，windows环境下只能=0

# 构造验证集
val_loader = DataLoader(dataset=val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0)

# -------------------------------------------------- #
# （3）数据可视化
# -------------------------------------------------- #
# 取出一个batch的训练集，返回图片及其标签
train_img, train_label = iter(train_loader).next()
# 查看shape, img=[32,3,224,224], label=[32]
print(train_img.shape, train_label.shape)

# 从一个batch中取出前9张图片
img = train_img[:9]  # [9, 3, 224, 224]
# 将图片反标准化，像素变到0-1之间
img = img / 2 + 0.5
# tensor类型变成numpy类型
img = img.numpy()
class_label = train_label.numpy()
# 维度重排 [b,c,h,w]==>[b,h,w,c]
img = np.transpose(img, [0, 2, 3, 1])

# 创建画板
plt.figure()
# 绘制四张图片
for i in range(img.shape[0]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(img[i])
    plt.xticks([])  # 不显示x轴刻度
    plt.yticks([])  # 不显示y轴刻度
    plt.title(class_names[class_label[i]])  # 图片对应的类别

plt.tight_layout()  # 轻量化布局
plt.show()

# -------------------------------------------------- #
# （4）加载模型
# -------------------------------------------------- #
# 1000分类层
net = resnet50(num_classes=1000, include_top=True)

# 加载预训练权重
net.load_state_dict(torch.load(weightpath, map_location=device))

# 为网络重写分类层
in_channel = net.fc.in_features  # 2048
net.fc = nn.Linear(in_channel, 4)  # [b,2048]==>[b,4]

# 将模型搬运到GPU上
net.to(device)
# 定义交叉熵损失
loss_function = nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.Adam(net.parameters(), lr=0.002)

# 保存准确率最高的一次迭代
best_acc = 0.0

# -------------------------------------------------- #
# （5）网络训练
# -------------------------------------------------- #
for epoch in range(epochs):

    print('-' * 30, '\n', 'epoch:', epoch)

    # 将模型设置为训练模型, dropout层和BN层只在训练时起作用
    net.train()

    # 计算训练一个epoch的总损失
    running_loss = 0.0

    # 每个step训练一个batch
    for step, data in enumerate(train_loader):
        # data中包含图像及其对应的标签
        images, labels = data

        # 梯度清零，因为每次计算梯度是一个累加
        optimizer.zero_grad()

        # 前向传播
        outputs = net(images.to(device))

        # 计算预测值和真实值的交叉熵损失
        loss = loss_function(outputs, labels.to(device))

        # 梯度计算
        loss.backward()

        # 权重更新
        optimizer.step()

        # 累加每个step的损失
        running_loss += loss.item()

        # 打印每个step的损失
        print(f'step:{step} loss:{loss}')

    # -------------------------------------------------- #
    # （6）网络验证
    # -------------------------------------------------- #
    net.eval()  # 切换为验证模型，BN和Dropout不起作用

    acc = 0.0  # 验证集准确率

    with torch.no_grad():  # 下面不进行梯度计算

        # 每次验证一个batch
        for data_test in val_loader:
            # 获取验证集的图片和标签
            test_images, test_labels = data_test

            # 前向传播
            outputs = net(test_images.to(device))

            # 预测分数的最大值
            predict_y = torch.max(outputs, dim=1)[1]

            # 累加每个step的准确率
            acc += (predict_y == test_labels.to(device)).sum().item()

        # 计算所有图片的平均准确率
        acc_test = acc / val_num

        # 打印每个epoch的训练损失和验证准确率
        print(f'total_train_loss:{running_loss / step}, total_test_acc:{acc_test}')

        # -------------------------------------------------- #
        # （7）权重保存
        # -------------------------------------------------- #
        # 保存最好的准确率的权重
        if acc_test > best_acc:
            # 更新最佳的准确率
            best_acc = acc_test
            # 保存的权重名称
            savename = savepath + 'resnet50.pth'
            # 保存当前权重
            torch.save(net.state_dict(), savename)