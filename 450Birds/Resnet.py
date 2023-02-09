# @Time    : 2023/2/7
# @Author  : Bing

import torch
import torch.nn as nn
import torchvision

net = torchvision.models.resnet50(num_classes=1000)
# print(net)

in_features = net.fc.in_features

net.fc = nn.Linear(in_features=in_features,out_features=450)
print(net)