# @Time    : 2023/2/22
# @Author  : Bing

import torch
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from MobileNetV3 import mobilenetv3  # 导入我们定义好了的模型文件
import numpy as np
import matplotlib.pyplot as plt

batch_size = 32
epochs = 10
best_loss = 2.0  # 当验证集损失小于2时再保存模型权重

filepath = 'E:/download/IDC/'
weightpath = 'E:/download/save_weight/mobilenet_v3_large-8738ca79.pth'
savepath = 'E:/download/save_weight/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)


# --------------------------------------------- #
# （1）数据集处理
# --------------------------------------------- #
# 定义预处理方法
data_transform = {
    # 训练集预处理
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机长宽比裁剪原始图片到224*224的大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 将numpy类型变成tensor类型，像素归一化，shape:[h,w,c]==>[c,h,w]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对图像的每个通道做标准化
    ]),

    # 验证集预处理
    'val': transforms.Compose([
        transforms.Resize((224, 224)),  # 将图像的大小缩放至224*224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# 图像导入并预处理
datasets = {
    'train': datasets.ImageFolder(filepath + 'training', transform=data_transform['train']),  # 读取训练集
    'val': datasets.ImageFolder(filepath + 'validation', transform=data_transform['val'])  # 读取验证集
}

# 构建数据集
dataloader = {
    'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True),  # 构造训练集
    'val': DataLoader(datasets['val'], batch_size=batch_size, shuffle=False)  # 构造验证集
}


# --------------------------------------------- #
# （2）查看数据集信息
# --------------------------------------------- #
train_num = len(datasets['train'])  # 查看训练集的图片数量
val_num = len(datasets['val'])  # 查看验证集的图片数量

# 查看分类类别及其索引 {0: 'begin', 1: 'malignant'}
LABEL = dict((v, k) for k, v in datasets['train'].class_to_idx.items())

# 查看训练集的简介
print(dataloader['train'].dataset)

# 从训练集中取出一个batch的图像及其标签
train_img, train_label = next(iter(dataloader['train']))
# 查看图像及标签的shape train_img.shape:[32, 3, 224, 224]  train_label.shape:[32]
print('train_img.shape: ', train_img.shape, 'train_label.shape:', train_label.shape)

# --------------------------------------------- #
# （3）数据可视化
# --------------------------------------------- #
# 从数据集中取出12张图片及其对应的标签
frame = train_img[:12]
frame_label = train_label[:12]

# 将图片从tensor类型变成numpy类型
frame = frame.numpy()

# 调整维度 [b,c,h,w]==>[b,h,w,c]
frame = np.transpose(frame, [0, 2, 3, 1])

# 对图像的反标准化
mean = [0.485, 0.456, 0.406]  # 均值
std = [0.229, 0.224, 0.225]  # 标准化
# 乘以标准差再加上均值
frame = frame * std + mean

# 将图像的像素值限制在0-1之间，小于0的取0，大于1的取1
frame = np.clip(frame, 0, 1)

# 绘制图像
plt.figure()
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(frame[i])  # 绘制单张图像
    plt.title(LABEL[frame_label[i].item()])  # 标签是图像的类别
    plt.axis('off')  # 不显示

plt.tight_layout()  # 轻量化布局
plt.show()

# --------------------------------------------- #
# （4）模型加载，迁移学习
# --------------------------------------------- #
# 接收模型，二分类
model = mobilenetv3(num_classes=2)
# 加载预训练权重文件，是字典类型。最后一层的神经元个数为1k
pre_weights = torch.load(weightpath, map_location=device)

# 遍历权重文件，保存除分类层以外的所有权重
pre_dict = {k: v for k, v in pre_weights.items() if 'classifier' not in k}
# len(pre_weights)  312
# len(pre_dict)  308

# 加载预训练权重，除了分类层以外其他都有预权重。
# 当strict=True,要求预训练权重层数的键值与新构建的模型中的权重层数名称完全吻合；
# 如果新构建的模型在层数上进行了部分微调，则上述代码就会报错：说key对应不上。
missing_keys, unexpected_keys = model.load_state_dict(pre_dict, strict=False)

# 冻结网络的倒残差结构的权重, model.parameters() 代表网络的所有参数
for param in model.inverted_block.parameters():
    param.requires_grad = False  # 参数不需要梯度更新

# --------------------------------------------- #
# （5）网络编译
# --------------------------------------------- #
# 将模型搬运至GPU上
model.to(device)
# 定义交叉熵损失
loss_function = nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------------------------- #
# （6）训练阶段
# --------------------------------------------- #
for epoch in range(epochs):
    # 打印当前训练轮次
    print('=' * 50, '\n', 'epoch: ', epoch)

    # 将模型设置为训练模式，dropout层和BN层起作用
    model.train()

    # 记录一个epoch的训练集总损失
    total_loss = 0.0

    # 每个step训练一个batch，包含数据集和标签
    for step, (images, labels) in enumerate(dataloader['train']):

        # 将数据集搬运到GPU上
        images, labels = images.to(device), labels.to(device)
        # 梯度清零，因为每次计算梯度是一个累加
        optimizer.zero_grad()
        # 前向传播，输出预测结果
        logits = model(images)

        # （1）计算损失
        # 计算每个step的预测值和真实值的交叉熵损失
        loss = loss_function(logits, labels)
        # 累加一个epoch中所有batch的损失
        total_loss += loss.item()

        # （2）反向传播
        # 梯度计算
        loss.backward()
        # 梯度更新
        optimizer.step()

        # 每100个batch打印一次当前的交叉熵损失
        if step % 100 == 0:
            print(f'step:{step}, train_loss:{loss}')

    # 计算一个epoch的平均损失，每个step的损失除以step的数量
    train_loss = total_loss / len(dataloader['train'])

    # --------------------------------------------- #
    # （7）验证训练
    # --------------------------------------------- #
    model.eval()  # 切换成验证模式，dropout和BN切换工作模式

    total_val_loss = 0.0  # 记录一个epoch的验证集损失
    total_val_correct = 0  # 记录一个epoch预测对了多少张图

    # 接下来不进行梯度更新
    with torch.no_grad():
        # 每个step测试一个batch
        for images, labels in dataloader['val']:
            # 将数据集搬运到GPU上
            images, labels = images.to(device), labels.to(device)
            # 前向传播 [b,c,h,w]==>[b,2]
            logits = model(images)

            # （1）损失计算
            # 计算每个batch的预测值和真实值的交叉熵损失
            loss = loss_function(logits, labels)
            # 累计每个batch的损失
            total_val_loss += loss.item()

            # （2）计算准确率
            # 找出每张图片的最大分数对应的索引，即每张图片对应什么类别
            pred = logits.argmax(dim=1)
            # 对比预测类别和真实类别，一个batch有多少个预测对了
            val_correct = torch.eq(pred, labels).float().sum()
            # 累加一个epoch中所有的batch被预测对的图片数量
            total_val_correct += val_correct

        # 计算一个epoch的验证集的平均损失和平均准确率
        val_loss = total_val_loss / len(dataloader['val'])  # 一个epoch中每个step的损失和除以step的总数
        val_acc = total_val_correct / val_num  # 一个epoch预测对的所有图片数量除以总图片数量

        # 打印一个epoch的训练集平均损失，验证集平均损失和准确率
        print('-' * 30)
        print(f'train_loss:{train_loss}, val_loss:{val_loss}, val_acc:{val_acc}')

        # --------------------------------------------- #
        # （8）保存权重
        # --------------------------------------------- #
        # 保存最小损失值对应的权重文件
        if val_loss < best_loss:
            # 权重文件名称
            savename = savepath + f'valacc{round(val_acc.item() * 100)}%_' + 'mobilenetv3.pth'
            # 保存该轮次的权重
            torch.save(model.state_dict(), savename)
            # 切换最小损失值
            best_loss = val_loss
            # 打印结果
            print(f'weights has been saved, best_loss has changed to {val_loss}')