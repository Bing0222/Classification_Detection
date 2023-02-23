# @Time    : 2023/2/22
# @Author  : Bing

import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from MobileNetV3 import mobilenetv3

from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# --------------------------------------------- #
# （0）参数设置
# --------------------------------------------- #
batch_size = 36  # 每批次处理72张图片

# 测试数据集地址
filepath = 'E:/download/IDC/testing'
# 模型训练权重文件位置
weightpath = "E:/download/save_weight/valacc83%_mobilenetv3.pth"

# 获取GPU设备，如果检测到GPU就用，没有就用CPU

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# --------------------------------------------- #
# （1）测试集数据处理
# --------------------------------------------- #
# 定义测试集预处理方法，和验证集的预处理方法相同
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 输入图像缩放至224*224
    transforms.ToTensor(),  # 转变数据类型，维度调整，归一化
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 每个通道的像素值标准化
])

# 加载测试集，并作预处理
datasets = datasets.ImageFolder(filepath, transform=data_transforms)

# 构造测试集
dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)

# 查看数据集信息 imgs.shape:[32, 3, 224, 224] labels.shape:[32]
test_images, test_labels = next(iter(dataloader))
print('imgs.shape:', test_images.shape, 'labels.shape:', test_labels.shape)

# 记录一共有多少张测试图片 72
test_num = len(datasets)

class_names = dict((v, k) for k, v in datasets.class_to_idx.items())


# --------------------------------------------- #
# （2）计算混淆矩阵值、精确率、召回率、F1
# --------------------------------------------- #
def metrics(logits, labels):
    # 计算每张图片对应的类别索引
    predict = logits.argmax(dim=1)

    # 计算混淆矩阵值，返回四个值 TN, FP, FN, TP
    cm = confusion_matrix(labels.cpu().numpy(), predict.cpu().numpy())

    # 获取 TN, FP, FN, TP
    tn, fp, fn, tp = cm.ravel()
    # 计算精确率
    precision = tp / (tp + fp)
    # 计算召回率
    recall = tp / (tp + fn)
    # 计算F1综合指标
    f1 = 2 * ((precision * recall) / (precision + recall))

    # 绘制混淆矩阵
    plt.figure()  # 创建画板
    plot_confusion_matrix(cm, figsize=(12, 8), cmap=plt.cm.Blues)  # 绘制混淆矩阵
    plt.xticks(range(2), list(class_names.values()), fontsize=14)  # x轴刻度名称
    plt.yticks(range(2), list(class_names.values()), fontsize=14)  # y轴刻度
    plt.xlabel('predict label', fontsize=16)  # x轴标签
    plt.ylabel('true label', fontsize=16)  # y轴标签
    plt.title(f'precision:{precision}, recall:{recall}, f1:{f1}')  # 标题
    #plt.show()

    return precision, recall, f1


# --------------------------------------------- #
# （3）模型构建
# --------------------------------------------- #
model = mobilenetv3(num_classes=2)
# 加载训练权重文件
model.load_state_dict(torch.load(weightpath, map_location=device))
# 将模型搬运至GPU上
model.to(device)

# 定义交叉熵损失
loss_function = nn.CrossEntropyLoss()

# 保存测试集的指标 precision, recall, f1
precisions = []
recalls = []
f1s = []

# --------------------------------------------- #
# （4）网络测试
# --------------------------------------------- #
model.eval()  # 切换成测试模式，改变BN和Dropout的工作模式

total_loss = 0.0  # 记录测试集总损失
test_correct = 0  # 记录测试集一共预测对了多少个

# 接下来的计算不需要更新梯度
with torch.no_grad():
    # 每次测试一个batch
    for step, (images, labels) in enumerate(dataloader):
        # 将数据集搬运到GPU上
        images, labels = images.to(device), labels.to(device)
        # 前向传播 [b,2]
        logits = model(images)

        # 计算每个batch的损失
        loss = loss_function(logits, labels)
        # 累加每个batch的测试损失
        total_loss += loss.item()

        # 计算每张图片对应的类别索引
        predict = logits.argmax(dim=1)
        # 对比预测结果和实际结果，比较预测对了多少张图片
        test_correct += torch.eq(predict, labels).float().sum()

        # 计算每个batch的评价指标，并绘制每个batch的混淆矩阵
        precision, recall, f1 = metrics(logits, labels)
        # 保存评价指标
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    # 计算平均准确率
    test_acc = test_correct / test_num

    # 打印测试集的总体损失和准确率
    print(f'total_loss:{avg_loss}, total_test_acc:{test_acc}')

    # 打印每个batch的评价指标
    print('batch_precision: ', precisions)
    print('batch_recalls: ', recalls)
    print('batch_f1s: ', f1s)