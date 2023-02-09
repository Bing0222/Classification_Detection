# @Time    : 2023/2/3
# @Author  : Bing


import torch
from torchvision import transforms
from PIL import Image
from ResNet import resnet50
import matplotlib.pyplot as plt

# -------------------------------------------------- #
# （0）参数设置
# -------------------------------------------------- #
# 图片文件路径
img_path = "E:/data/brids/predict_data/bird_photos/bird1.jpg"
# 权重参数路径
weights_path = "E:/data/brids/weights/resnet50.pth"
# 预测索引对应的类别名称
class_names = ['Bananaquit', 'Black Skimmer', 'Black Throated Bushtiti', 'Cockatoo']

# 获取GPU设备
if torch.cuda.is_available():  # 如果有GPU就用，没有就用CPU
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# -------------------------------------------------- #
# （1）数据加载
# -------------------------------------------------- #
# 预处理函数
data_transform = transforms.Compose([
    # 将输入图像的尺寸变成224*224
    transforms.Resize((224, 224)),
    # 数据变成tensor类型，像素值归一化，调整维度[h,w,c]==>[c,h,w]
    transforms.ToTensor(),
    # 对每个通道的像素进行标准化，给出每个通道的均值和方差
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# 读取图片
frame = Image.open(img_path)
# 展示图片
plt.imshow(frame)
plt.title('Black_Throated_Bushtiti')
plt.show()

# 数据预处理
img = data_transform(frame)
# 给图像增加batch维度 [c,h,w]==>[b,c,h,w]
img = torch.unsqueeze(img, dim=0)

# -------------------------------------------------- #
# （2）图像预测
# -------------------------------------------------- #
# 加载模型
model = resnet50(num_classes=4, include_top=True)
# 加载权重文件
model.load_state_dict(torch.load(weights_path, map_location=device))
# 模型切换成验证模式，dropout和bn切换形式
model.eval()

# 前向传播过程中不计算梯度
with torch.no_grad():
    # 前向传播
    outputs = model(img)
    # 只有一张图就挤压掉batch维度
    outputs = torch.squeeze(outputs)
    # 计算图片属于4个类别的概率
    predict = torch.softmax(outputs, dim=0)
    # 得到类别索引
    predict_cla = torch.argmax(predict).numpy()

# 获取最大预测类别概率
predict_score = round(torch.max(predict).item(), 4)
# 获取预测类别的名称
predict_name = class_names[predict_cla]

# 展示预测结果
plt.imshow(frame)
plt.title('class: ' + str(predict_name) + '\n score: ' + str(predict_score))
plt.show()
