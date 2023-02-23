import torch
from torch import nn
from torchstat import stat  # 查看网络参数

# ---------------------------------------------------- #
# （2）标准卷积模块
'''
in_channel：输入特征图的通道数
out_channel： 卷积输出的通道数
kernel_size： 卷积核尺寸
stride： 卷积的步长
activation：'RE'和'HS'，使用RELU激活还是HardSwish激活
'''


# ---------------------------------------------------- #
class conv_block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, activation):
        super(conv_block, self).__init__()

        # 普通卷积
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2, bias=False)
        # BN标准化
        self.bn = nn.BatchNorm2d(num_features=out_channel)

        # 使用何种激活函数
        if activation == 'RE':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'HS':
            self.act = nn.Hardswish(inplace=True)

    # 前向传播
    def forward(self, inputs):

        # 卷积+BN+激活
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        return x


# ---------------------------------------------------- #
# （3）SE注意力机制
'''
in_channel：代表输入特征图的通道数
ratio：第一个全连接层下降的通道数
'''


# ---------------------------------------------------- #
class se_block(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super(se_block, self).__init__()

        # 全局平均池化, [b,c,h,w]==>[b,c,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # 第一个全连接层，将通道数下降为原来的四分之一
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        # relu激活函数
        self.relu = nn.ReLU()
        # 第二个全连接层，恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)
        # hard_sigmoid激活函数，通道权值归一化
        self.hsigmoid = nn.Hardsigmoid()

    # 前向传播
    def forward(self, inputs):
        # 获取输入图像的shape
        b, c, h, w = inputs.shape
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(inputs)
        # 维度调整 [b,c,1,1]==>[b,c]
        x = x.view([b, c])
        # 第一个全连接下降通道 [b,c]==>[b,c//4]
        x = self.fc1(x)
        # relu激活
        x = self.relu(x)
        # 第二个全连接恢复通道 [b,c//4]==>[b,c]
        x = self.fc2(x)
        # sigmoid权值归一化
        x = self.hsigmoid(x)
        # 维度调整 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])
        # 将输入图像和归一化由的通道权值相乘
        outputs = inputs * x

        return outputs


# ---------------------------------------------------- #
# （4）倒残差结构
'''
in_channel：输入特征图的通道数
expansion： 第一个1*1卷积上升的通道数
out_channel： 最后一个1*1卷积下降的通道数
kernel_size： 深度可分离卷积的卷积核尺寸
stride： 深度可分离卷积的步长
se： 布尔类型，是否再深度可分离卷积之后使用通道注意力机制
activation：'RE'和'HS'，使用RELU激活还是HardSwish激活
'''


# ---------------------------------------------------- #
class InvertedResBlock(nn.Module):
    # 初始化
    def __init__(self, in_channel, kernel_size, expansion, out_channel, se, activation, stride):
        # 继承父类初始化方法
        super(InvertedResBlock, self).__init__()

        # 属性分配
        self.stride = stride
        self.expansion = expansion

        # 1*1卷积上升通道数
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=expansion, kernel_size=1,
                               stride=1, padding=0, bias=False)

        # 标准化，传入特征图的通道数
        self.bn1 = nn.BatchNorm2d(num_features=expansion)

        # 3*3深度卷积提取特征, groups代表将输入特征图分成多少组，groups=expansion使卷积核的个数和输入特征图相同
        self.conv2 = nn.Conv2d(in_channels=expansion, out_channels=expansion, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, bias=False, groups=expansion)

        # 标准化
        self.bn2 = nn.BatchNorm2d(num_features=expansion)

        # 1*1卷积下降通道数
        self.conv3 = nn.Conv2d(in_channels=expansion, out_channels=out_channel, kernel_size=1,
                               stride=1, padding=0, bias=False)

        # 标准化
        self.bn3 = nn.BatchNorm2d(num_features=out_channel)

        # 激活函数的选择
        if activation == 'RE':  # relu激活函数
            self.act = nn.ReLU(inplace=True)
        elif activation == 'HS':  # hard_swish激活函数
            self.act = nn.Hardswish(inplace=True)

        # 是否使用SE注意力机制
        if se is True:  # 对深度卷积的输出特征图使用通道注意力机制
            self.se_block = se_block(in_channel=expansion)
        else:
            self.se_block = nn.Identity()  # 如果不做SE那么输入等于输出，不做变换

    # 前向传播
    def forward(self, x):
        # 获取输入图像的shape
        b, c, h, w = x.shape

        # 残差边部分
        residual = x

        # 如果输入图像的channel和第一个1*1卷积上升的通道数相同，那么就不需要做1*1卷积升维
        if c != self.expansion:
            # 1*1卷积+BN+激活
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act(x)

        # 3*3深度卷积提取特征输入和输出通道数相同
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        # 使用注意力机制，或者不使用（该模块的输入等于输出）
        x = self.se_block(x)

        # 1*1卷积下降通道数
        x = self.conv3(x)
        x = self.bn3(x)

        # 如果深度卷积的步长等于1并且输入和输出的shape相同，就用残差连接输入和输出
        if self.stride == 1 and residual.shape == x.shape:
            outputs = x + residual
        # 否则就直接输出下采样后的结果
        else:
            outputs = x

        return outputs


# ---------------------------------------------------- #
# （5）主干网络
# ---------------------------------------------------- #
class mobilenetv3(nn.Module):
    # 初始化num_classes代表最终的分类数, width_mult代表宽度因子
    def __init__(self, num_classes, width_mult=1.0):
        super(mobilenetv3, self).__init__()

        # 第一个下采样卷积层 [b,3,224,224]==>[b,16,112,112]
        self.conv_block1 = conv_block(in_channel=3, out_channel=16, kernel_size=3, stride=2, activation='HS')

        # 倒残差结构
        inverted_block = [

            # in_channel, kernel_size, expansion, out_channel, se, activation, stride
            InvertedResBlock(16, 3, 16, 16, False, 'RE', 1),
            InvertedResBlock(16, 3, 64, 24, False, 'RE', 2),  # [b,16,112,112]==>[b,24,56,56]
            InvertedResBlock(24, 3, 72, 24, False, 'RE', 1),
            InvertedResBlock(24, 5, 72, 40, True, 'RE', 2),  # [b,24,56,56]==>[b,40,28,28]
            InvertedResBlock(40, 5, 120, 40, True, 'RE', 1),
            InvertedResBlock(40, 5, 120, 40, True, 'RE', 1),
            InvertedResBlock(40, 3, 240, 80, False, 'HS', 2),  # [b,40,28,28]==>[b,80,14,14]
            InvertedResBlock(80, 3, 200, 80, False, 'HS', 1),
            InvertedResBlock(80, 3, 184, 80, False, 'HS', 1),
            InvertedResBlock(80, 3, 184, 80, False, 'HS', 1),
            InvertedResBlock(80, 3, 480, 112, True, 'HS', 1),
            InvertedResBlock(112, 3, 672, 112, True, 'HS', 1),
            InvertedResBlock(112, 5, 672, 160, True, 'HS', 1),
            InvertedResBlock(160, 5, 672, 160, True, 'HS', 2),  # [b,80,14,14]==>[b,160,7,7]
            InvertedResBlock(160, 5, 960, 160, True, 'HS', 1),
        ]

        # 将堆叠的倒残差结构以非关键字参数返回
        self.inverted_block = nn.Sequential(*inverted_block)

        # 1*1卷积调整通道 [b,160,7,7]==>[b,960,7,7]
        self.conv_block2 = conv_block(in_channel=160, out_channel=960,
                                      kernel_size=1, stride=1, activation='HS')

        # 全局平均池化 ==> [b,960,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 分类层，先用一个全连接调整通道，再用一个全连接分类
        self.classify = nn.Sequential(
            # [b,960]==>[b,1280]
            nn.Linear(in_features=960, out_features=1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2, inplace=True),
            # [b,1280]==>[b,num_classes]
            nn.Linear(in_features=1280, out_features=num_classes))

        # 权值初始化
        for m in self.modules():
            # 对卷积层使用kaiming初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                # 对偏置初始化
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # 对标准化层初始化
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            # 对全连接层初始化
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    # 前向传播
    def forward(self, inputs):

        # [b,3,224,224]==>[b,16,112,112]
        x = self.conv_block1(inputs)
        # [b,16,112,112]==>[b,160,7,7]
        x = self.inverted_block(x)
        # [b,160,7,7]==>[b,960,7,7]
        x = self.conv_block2(x)
        # [b,960,7,7]==>[b,960,1,1]
        x = self.avg_pool(x)
        # 展平去除宽高维度 [b,960,1,1]==>[b,960]
        x = torch.flatten(x, 1)
        # [b,960]==>[b,num_classes]
        x = self.classify(x)

        return x


# ---------------------------------------------------- #
# （6）查看网络结构
# ---------------------------------------------------- #
if __name__ == '__main__':
    # 模型实例化
    model = mobilenetv3(num_classes=1000)
    # 构造输入层shape==[4,3,224,224]
    inputs = torch.rand(4, 3, 224, 224)

    # 前向传播查看输出结果
    outputs = model(inputs)
    print(outputs.shape)  # [4, 1000]

    # 查看模型参数，不需要指定batch维度
    stat(model, input_size=[3, 224, 224])

    # Total params: 5,140,608
    # Total memory: 44.65MB
    # Total MAdd: 505.77MMAdd
    # Total Flops: 255.62MFlops
    # Total MemR+W: 96.79MB
