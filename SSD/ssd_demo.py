# @Time    : 2023/3/17
# @Author  : Bing

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



def xy_to_cxcy(xy):
    """
    边界坐标转中心坐标
    :param xy: 一个shape为 [n,4]的tensor，表示n个边界坐标
    :return: 一个shape为 [n,4]的tensor，表示转换后的n个中心坐标
    """
    return torch.cat([
        (xy[:, 2:] + xy[:, :2]) / 2,  # cx, cy
        xy[:, 2:] - xy[:, :2]  # w, h
    ], dim=1)


def cxcy_to_xy(cxcy):
    """
    中心坐标转边界坐标
    :param cxcy: 一个shape为 [n,4]的tensor，表示n个中心坐标
    :return: 一个shape为 [n,4]的tensor，表示转换后的n个边界坐标
    """
    return torch.cat([
        cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
        cxcy[:, :2] + (cxcy[:, 2:] / 2)  # x_max, y_max
    ], dim=1)


def find_intersection(set_1, set_2):
    """
    计算第一个集合中每个框与第二个集合中每个框的交集面积

    :param set_1: 一个shape为[m,4]的tensor，代表m个边界坐标
    :param set_2: 一个shape为[n,4]的tensor，代表n个边界坐标
    :return: 一个shape为[m,n]的tensor，例如：[0,:]表示set_1中第1个框与set_2中每个框的交集面积
    """
    # max函数中的两个tensor的shape分别为[m,1,2], [1,n,2],可以应用广播机制，最后得到的tensor的shape为[m,n,2]
    # 例如：[0, :, 2]表示set_1中第一个框与set_2中所有框交集的左上角坐标
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # [m, n, 2]
    # 计算右下角的坐标
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # [m, n, 2]
    # 将两个减式小于0的设置为0
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # [m, n, 2]
    # 相乘得到交集面积
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # [m, n]


def find_jaccard_overlap(set_1, set_2):
    """
    计算第一个集合中每个框与第二个集合中每个框的Jaccard系数

    :param set_1: 一个shape为[m,4]的tensor，代表m个边界坐标
    :param set_2: 一个shape为[n,4]的tensor，代表n个边界坐标
    :return: 一个shape为[m,n]的tensor，例如：[0,:]表示set_1中第1个框与set_2中每个框的Jaccard系数
    """
    # 每个框与其他框的交集
    intersection = find_intersection(set_1, set_2)  # [m, n]

    # 计算每个集合中每个框的面积
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # [m]
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # [n]

    # 总面积减去交集就是并集
    # unsqueeze的作用同样是为了满足广播机制的条件
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # [m, n]
    # Jaccard系数 = 交集面积 / 并集面积
    return intersection / union  # [m, n]


def decimate(tensor, interval=None):
    """
    根据间隔点，对tensor进行采样
    :param tensor: 一个有n个维度，需要进行采样的tensor
    :param interval: 一个元素为n的列表，每个位置的元素表示该维度的采样步长
    :return: 对每个维度采样后的n维tensor
    """
    # tensor的维度必须和采样的维度数量一样
    if tensor.dim() != len(interval):
        raise ValueError('tensor and interval must have same dimensions !')

    for d in range(tensor.dim()):
        # 如果为None则不对该维度进行采样
        if interval[d] is not None:
            # 根据步长进行采样
            tensor = tensor.index_select(
                dim=d,
                index=torch.arange(start=0, end=tensor.size(d), step=interval[d]).long()
            )
    return tensor


class VGGBase(nn.Module):
    def __init__(self):
        super(VGGBase, self).__init__()
        # 标准VGG16卷积层
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # 第三个最大池化层需要设置 ceil 模式
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        # 第五个最大池化层的参数需要进行修改
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # 原VGG16的fc6和fc7层，需要从全连接层转换为卷积层，同时删除部分权重
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=6, dilation=(6, 6))
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
        # 为每个卷积层设置训练好的权重
        self.load_weights()

    def load_weights(self):
        """
        为卷积层加载训练好的参数，同时对全连接层进行转化和采样
        """
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())
        # 获取预训练权重
        pretrained_state_dict = vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        # 除了fc6和fc7层之外，其他的层直接加载权重
        for i, param in enumerate(param_names[:-4]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        def decimate(tensor, interval=None):
            """
            根据间隔点，对tensor进行采样
            :param tensor: 一个有n个维度，需要进行采样的tensor
            :param interval: 一个元素为n的列表，每个位置的元素表示该维度的采样步长
            :return: 对每个维度采样后的n维tensor
            """
            # tensor的维度必须和采样的维度数量一样
            if tensor.dim() != len(interval):
                raise ValueError('tensor and interval must have same dimensions !')
            for d in range(tensor.dim()):
                # 如果为None则不对该维度进行采样
                if interval[d] is not None:
                    # 根据步长进行采样
                    tensor = tensor.index_select(
                        dim=d,
                        index=torch.arange(start=0, end=tensor.size(d), step=interval[d]).long()
                    )
            return tensor

        # 将fc6的权重reshape为 4096,7,7,512。但是pytorch的通道数在前面，因此reshape为 4096,512,7,7
        fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)
        # fc6层偏置的权重
        fc6_bias = pretrained_state_dict['classifier.0.bias']
        # 对权重进行采样，[4096, 512, 7, 7] -> [1024, 512, 3, 3]
        state_dict['conv6.weight'] = decimate(fc6_weight, [4, None, 3, 3])
        # 对偏置进行同样的采样，[4096] -> [1024]
        state_dict['conv6.bias'] = decimate(fc6_bias, [4])

        # fc7层与fc6层操作一样
        fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)
        fc7_bias = pretrained_state_dict['classifier.3.bias']
        # [4096, 4096, 1, 1] -> [1024, 1024, 1, 1]
        state_dict['conv7.weight'] = decimate(fc7_weight, [4, 4, None, None])
        # [4096] -> [1024]
        state_dict['conv7.bias'] = decimate(fc7_bias, [4])

        del decimate
        # 加载权重
        self.load_state_dict(state_dict)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))  # [b, 3, 300, 300] -> [b, 64, 300, 300]
        x = F.relu(self.conv1_2(x))  # [b, 64, 300, 300] -> [b, 64, 300, 300]
        x = self.pool1(x)  # [b, 64, 300, 300] -> [b, 64, 150, 150]

        x = F.relu(self.conv2_1(x))  # [b, 64, 150, 150] -> [b, 128, 150, 150]
        x = F.relu(self.conv2_2(x))  # [b, 128, 150, 150] -> [b, 128, 150, 150]
        x = self.pool2(x)  # [b, 128, 150, 150] -> [b, 128, 75, 75]

        x = F.relu(self.conv3_1(x))  # [b, 128, 75, 75] -> [b, 256, 75, 75]
        x = F.relu(self.conv3_2(x))  # [b, 256, 75, 75] -> [b, 256, 75, 75]
        x = F.relu(self.conv3_3(x))  # [b, 256, 75, 75] -> [b, 256, 75, 75]
        x = self.pool3(x)  # [b, 256, 75, 75] -> [b, 256, 38, 38]

        x = F.relu(self.conv4_1(x))  # [b, 256, 38, 38] -> [b, 512, 38, 38]
        x = F.relu(self.conv4_2(x))  # [b, 512, 38, 38] -> [b, 512, 38, 38]
        conv4_3_feats = F.relu(self.conv4_3(x))  # [b, 512, 38, 38] -> [b, 512, 38, 38]
        x = self.pool4(conv4_3_feats)  # [b, 512, 38, 38] -> [b, 512, 19, 19]

        x = F.relu(self.conv5_1(x))  # [b, 512, 19, 19] -> [b, 512, 19, 19]
        x = F.relu(self.conv5_2(x))  # [b, 512, 19, 19] -> [b, 512, 19, 19]
        x = F.relu(self.conv5_3(x))  # [b, 512, 19, 19] -> [b, 512, 19, 19]
        x = self.pool5(x)  # [b, 512, 19, 19] -> [b, 512, 19, 19]

        x = F.relu(self.conv6(x))  # [b, 512, 19, 19] -> [b, 1024, 19, 19]
        conv7_feats = F.relu(self.conv7(x))  # [b, 1024, 19, 19] -> [b, 1024, 19, 19]
		# 注意这里的输出有两个
        return conv4_3_feats, conv7_feats


class AuxiliaryConvolutions(nn.Module):
    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
        # 初始化权重
        self.init_weight()

    def init_weight(self):
        for conv in self.children():
            if isinstance(conv, nn.Conv2d):
                nn.init.xavier_normal_(conv.weight)
                nn.init.constant_(conv.bias, 0.)

    def forward(self, x):
        # 这里的参数x是基础卷积的conv7层输出
        x = F.relu(self.conv8_1(x))  # [b, 1024, 19, 19] -> [b, 256, 19, 19]
        conv8_2_feats = F.relu(self.conv8_2(x))  # [b, 256, 19, 19] -> [b, 512, 10, 10]

        x = F.relu(self.conv9_1(conv8_2_feats))  # [b, 512, 10, 10] -> [b, 128, 10, 10]
        conv9_2_feats = F.relu(self.conv9_2(x))  # [b, 128, 10, 10] -> [b, 256, 5, 5]

        x = F.relu(self.conv10_1(conv9_2_feats))  # [b, 256, 5, 5] -> [b, 128, 5, 5]
        conv10_2_feats = F.relu(self.conv10_2(x))  # [b, 128, 5, 5] -> [b, 256, 3, 3]

        x = F.relu(self.conv11_1(conv10_2_feats))  # [b, 256, 3, 3] -> [b, 128, 3, 3]
        conv11_2_feats = F.relu(self.conv11_2(x))  # [b, 128, 3, 3] -> [b, 256, 1, 1]

        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats


@staticmethod
def create_prior_boxes():
    # 特征图的尺寸
    features_dim = {'conv4_3': 38, 'conv7': 19, 'conv8_2': 10,
                    'conv9_2': 5, 'conv10_2': 3, 'conv11_2': 1}
    # prior的scale
    object_scales = {'conv4_3': 0.1, 'conv7': 0.2, 'conv8_2': 0.375,
                     'conv9_2': 0.55, 'conv10_2': 0.725, 'conv11_2': 0.9}
    # prior的aspect ratio
    # conv7，conv8_2和conv9_2会多出 3:1 和 1:3
    aspect_ratios = {
        'conv4_3': [1., 2., 0.5],
        'conv7': [1., 2., 3., 0.5, 0.333],
        'conv8_2': [1., 2., 3., 0.5, 0.333],
        'conv9_2': [1., 2., 3., 0.5, 0.333],
        'conv10_2': [1., 2., 0.5],
        'conv11_2': [1., 2., 0.5]
    }
    # 记录特征图的名称，用来查找当前特征图的下一个特征图
    features_name = list(features_dim.keys())
    # 所有的priors
    prior_boxes = []
    # 每个特征图都会有priors
    for k, feature in enumerate(features_name):
        # 每个特征图的每个位置都有priors
        # 模仿卷积的操作，按照从左到右，从上到下的顺序计算priors，为了与预测卷积相匹配
        for i in range(features_dim[feature]):
            for j in range(features_dim[feature]):
                # 当前特征图的当前格子的中心坐标（需要进行缩放）
                cx = (j + 0.5) / features_dim[feature]
                cy = (i + 0.5) / features_dim[feature]
                # 为当前格子按照aspect ratios生成priors
                for ratio in aspect_ratios[feature]:
                    # w = s * sqrt(a)， h = s / sqrt(a)
                    # 计算每个prior的中心坐标
                    prior_boxes.append([cx, cy, object_scales[feature] * sqrt(ratio),
                                        object_scales[feature] / sqrt(ratio)])
                    # 当ratio时，需要额外添加一个prior
                    if ratio == 1.:
                        # 如果当前特征图不是最后一个特征图，即当前特征图不是conv11_2
                        if k != len(features_name) - 1:
                            # 那么这个额外的prior的scale就是 sqrt(当前特征图scale * 下一个特征图scale)
                            additional_scale = sqrt(object_scales[feature] * object_scales[features_name[k + 1]])
                        else:
                            # 如果当前特征图是最后一个，它就不存在下一个特征图，直接将scale设置为1
                            additional_scale = 1.
                        # 添加额外的prior的中心坐标
                        prior_boxes.append([cx, cy, additional_scale, additional_scale])
    # 最后将所有priors转换为一个tensor
    prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # [8732, 4]
    return prior_boxes


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    将对应的两组框转换为之间的偏移量
    :param cxcy:  维度为 [n, 4] 的tensor，表示一组框的中心坐标
    :param priors_cxcy: 维度为 [n, 4]的tensor，表示一组框的中心坐标
    :return: 返回对应两个框之间的偏移量，维度为 [n, 4]
    """
    # 10和5是为了缩放梯度，完全是经验性的
    return torch.cat([
        (cxcy[:, :2] - priors_cxcy[:, :2]) / priors_cxcy[:, 2:] * 10,  # g_cx, g_cy
        torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5  # g_w, g_h
    ], dim=1)

def gcxcy_to_cxcy(gcxcy, priors_cxcy):
    """
    偏移量和对应的框，转为偏移后的框
    :param gcxcy:  维度为 [n, 4] 的tensor，表示偏移量
    :param priors_cxcy: 维度为 [n, 4]的tensor，表示框的中心坐标
    :return: 返回偏移后的框，维度为 [n, 4]
    """
    return torch.cat([
        gcxcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # cx,cy
        torch.exp(gcxcy[:, 2:] / 5) * priors_cxcy[:, 2:]  # w, h
    ], dim=1)


class PredictionConvolutions(nn.Module):
    def __init__(self, n_classes):
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        # 每个特征层上每一个点所设置的先验框个数
        n_priors_boxes = {'conv4_3': 4, 'conv7': 6, 'conv8_2': 6,
                          'conv9_2': 6, 'conv10_2': 4, 'conv11_2': 4}

        # 位置预测卷积
        self.loc_conv4_3 = nn.Conv2d(512, n_priors_boxes['conv4_3'] * 4, kernel_size=(3, 3), padding=1)
        self.loc_conv7 = nn.Conv2d(1024, n_priors_boxes['conv7'] * 4, kernel_size=(3, 3), padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, n_priors_boxes['conv8_2'] * 4, kernel_size=(3, 3), padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, n_priors_boxes['conv9_2'] * 4, kernel_size=(3, 3), padding=1)
        self.loc_conv10_2 = nn.Conv2d(256, n_priors_boxes['conv10_2'] * 4, kernel_size=(3, 3), padding=1)
        self.loc_conv11_2 = nn.Conv2d(256, n_priors_boxes['conv11_2'] * 4, kernel_size=(3, 3), padding=1)

        # 类别预测卷积
        self.class_conv4_3 = nn.Conv2d(512, n_priors_boxes['conv4_3'] * n_classes, kernel_size=(3, 3), padding=1)
        self.class_conv7 = nn.Conv2d(1024, n_priors_boxes['conv7'] * n_classes, kernel_size=(3, 3), padding=1)
        self.class_conv8_2 = nn.Conv2d(512, n_priors_boxes['conv8_2'] * n_classes, kernel_size=(3, 3), padding=1)
        self.class_conv9_2 = nn.Conv2d(256, n_priors_boxes['conv9_2'] * n_classes, kernel_size=(3, 3), padding=1)
        self.class_conv10_2 = nn.Conv2d(256, n_priors_boxes['conv10_2'] * n_classes, kernel_size=(3, 3), padding=1)
        self.class_conv11_2 = nn.Conv2d(256, n_priors_boxes['conv11_2'] * n_classes, kernel_size=(3, 3), padding=1)

        # 初始化权重
        self.init_weight()

    def init_weight(self):
        for conv in self.children():
            if isinstance(conv, nn.Conv2d):
                nn.init.xavier_normal_(conv.weight)
                nn.init.constant_(conv.bias, 0.)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
        batch_size = conv4_3_feats.size(0)

        # 预测框的边界
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # [b, 512, 38, 38] -> [b, 16, 38, 38]
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()  # [b, 16, 38, 38] -> [b, 38, 38, 16]
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)  # [b, 38, 38, 16] -> [b, 5776, 4]

        l_conv7 = self.loc_conv7(conv7_feats)  # [b, 1024, 19, 19] -> [b, 24, 19, 19]
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()  # [b, 24, 19, 19] -> [b, 19, 19, 24]
        l_conv7 = l_conv7.view(batch_size, -1, 4)  # [b, 19, 19, 24] -> [b, 2166, 4]

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # [b, 512, 10, 10] -> [b, 24, 10, 10]
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()  # [b, 24, 10, 10] -> [b, 10, 10, 24]
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  # [b, 10, 10, 24] -> [b, 600, 4]

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  # [b, 256, 5, 5] -> [b, 24, 5, 5]
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()  # [b, 24, 5, 5] -> [b, 5, 5, 24]
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)  # [b, 5, 5, 24] -> [b, 150, 4]

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  # [b, 256, 3, 3] -> [b, 16, 3, 3]
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()  # [b, 16, 3, 3] -> [b, 3, 3, 16]
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)  # [b, 3, 3, 16] -> [b, 36, 4]

        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)  # [b, 256, 1, 1] -> [b, 16, 1, 1]
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()  # [b, 16, 1, 1] -> [b, 1, 1, 16]
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)  # [b, 1, 1, 16] -> [b, 4, 4]

        # 预测框的类别

        # [b, 512, 38, 38] -> [b, 4 * n_classes, 38, 38]
        c_conv4_3 = self.class_conv4_3(conv4_3_feats)
        # [b, 4 * n_classes, 38, 38] -> [b, 38, 38, 4 * n_classes]
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()
        # [b, 38, 38, 4 * n_classes] -> [b, 5776, n_classes]
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.n_classes)

        # [b, 1024, 19, 19] -> [b, 6 * n_classes, 19, 19]
        c_conv7 = self.class_conv7(conv7_feats)
        # [b, 6 * n_classes, 19, 19] -> [b, 19, 19, 6 * n_classes]
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()
        # [b, 19, 19, 6 * n_classes] -> [b, 2166, n_classes]
        c_conv7 = c_conv7.view(batch_size, -1, self.n_classes)

        # [b, 512, 10, 10] -> [b, 6 * n_classes, 10, 10]
        c_conv8_2 = self.class_conv8_2(conv8_2_feats)
        # [b, 6 * n_classes, 10, 10] -> [b, 10, 10, 6 * n_classes]
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()
        # [b, 10, 10, 6 * n_classes] -> [b, 600, n_classes]
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)

        # [b, 256, 5, 5] -> [b, 6 * n_classes, 5, 5]
        c_conv9_2 = self.class_conv9_2(conv9_2_feats)
        # [b, 6 * n_classes, 5, 5] -> [b, 5, 5, 6 * n_classes]
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()
        # [b, 5, 5, 6 * n_classes] -> [b, 150, n_classes]
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)

        # [b, 256, 3, 3] -> [b, 4 * n_classes, 3, 3]
        c_conv10_2 = self.class_conv10_2(conv10_2_feats)
        # [b, 4 * n_classes, 3, 3] -> [b, 3, 3, 4 * n_classes]
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()
        # [b, 3, 3, 4 * n_classes] -> [b, 36, n_classes]
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes)

        # [b, 256, 1, 1] -> [b, 4 * n_classes, 1, 1]
        c_conv11_2 = self.class_conv11_2(conv11_2_feats)
        # [b, 4 * n_classes, 1, 1] -> [b, 1, 1, 4 * n_classes]
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()
        # [b, 1, 1, 4 * n_classes] -> [b, 4, n_classes]
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)

        # [b, 8732, 4]
        loc = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)
        # [b, 8732, n_classes]
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2], dim=1)

        return loc, classes_scores


class MultiBoxLoss(nn.Module):
    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        """
        物体检测的损失函数
        :param priors_cxcy:     tensor，默认生成的priors，中心坐标形式
        :param threshold:       标量，表示设定重叠程度的阈值，当Jaccard系数大于阈值时认为是正匹配，默认为0.5
        :param neg_pos_ratio:   标量，表示采样的负样本与正样本的比例，默认为3
        :param alpha:           标量，表示将定位损失和分类损失以什么比例相加，默认为1
        """
        super(MultiBoxLoss, self).__init__()

        self.priors_cxcy = priors_cxcy
        # priors的边界坐标表示
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predicted_loc, predicted_scores, boxes, labels):
        """
        前向计算过程
        :param predicted_loc:      SSD300模型预测的位置，[b, 8732, 4]
        :param predicted_scores:   SSD300模型预测的类别分数 [b, 8732, n_classes]
        :param boxes:              真实框 [b, n_objects, 4]，注意n_objects不是固定数值，每张图片内的物体个数可能不一样
        :param labels:             真实标签 [b, n_objects]
        :return:                   标量，代表损失
        """
        batch_size = predicted_loc.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_loc.size(1) == predicted_scores.size(1)

        true_loc = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # [b, 8732, 4]
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # [b, 8732]

        # 对每张图片
        for i in range(batch_size):
            n_objects = boxes[i].size(0)
            # 计算先验框与真实框的Jaccard系数
            overlap = find_jaccard_overlap(boxes[i], self.priors_xy)

            # 对于每个先验框，找到具有最大重叠的对象
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)

            # 我们不希望遇到这样的情况：存在某个物体没有被正先验框所表示，这包含两种情况
            # 1. 对每个先验框，我们选择其与真实框重叠最大的那个物体作为最佳检测物体，这可能导致某个物体没有一个先验框与之对应
            # 2. 对于有匹配物体的先验框来说，如果其重叠程度低于设定的阈值（0.5），也将被设置为背景类

            # 首先找到每个物体所对应的重叠程度最大的先验框
            _, prior_for_each_object = overlap.max(dim=1)
            # 然后将每个物体分配给相应的具有最大重叠的先验框，这解决了第1种情况
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)
            # 为了保证这些先验框合格，人为赋予一个大于阈值（0.5）的值，这解决了第2种情况
            overlap_for_each_prior[prior_for_each_object] = 1.

            # 每个先验框的标签
            label_for_each_prior = labels[i][object_for_each_prior]
            # 将重叠程度小于阈值的标签设置为0（背景类）
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0
            true_classes[i] = label_for_each_prior
            # 将真实框编码为我们预测的偏移量形式，[8732, 4]
            true_loc[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)
        positive_priors = true_classes != 0
        # 仅在正先验条件下计算定位损失
        loc_loss = self.smooth_l1(predicted_loc[positive_priors], true_loc[positive_priors])

        # 置信度损失是在每个图片上的正先验和最困难的负先验上计算的
        n_positives = positive_priors.sum(dim=1)
        # 我们将使用最困难的（neg_pos_ratio * n_positives 个）负先验，拥有最大的loss
        # 这叫做硬负采样，它专注于每张图片上最困难的负先验，同时也最大限度的减少了正负样本不均衡问题
        n_hard_negatives = self.neg_pos_ratio * n_positives

        # 首先计算所有先验的损失
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # [b * 8732]
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # [b, 8732]

        # 我们已经知道哪些先验是正的
        conf_loss_pos = conf_loss_all[positive_priors]

        # 接着，我们需要寻找最困难的先验
        # 为了实现目标，我们仅根据每张图片上的负先验按照其loss的降序排列，然后取前 n_hard_negatives 个，作为最困难的负先验
        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[positive_priors] = 0.  # 将正先验设置为0，这样按照降序排序的时候，负先验会在前面
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # [b, 8732]
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # [b, 8732]
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]
        # 像论文中一样，仅在正先验上求平均，尽管正先验和负先验都进行了计算
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()

        # 返回总损失
        return conf_loss + self.alpha * loc_loss


class SSD300(nn.Module):
    def __init__(self, n_classes):
        super(SSD300, self).__init__()

        self.n_classes = n_classes

        self.base = VGGBase()  # 基础卷积
        self.aux_convs = AuxiliaryConvolutions()  # 辅助卷积
        self.pred_convs = PredictionConvolutions(n_classes)  # 预测卷积

        # 我们认为低级特征有很大的规模，因此使用 L2范数 重新进行缩放， 这是一个可训练参数
        # conv4_3_feats 有512个channels
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        nn.init.constant_(self.rescale_factors, 20)

        # 先验框 priors
        self.prior_cxcy_boxes = self.create_prior_boxes()

    def forward(self, x):
        # x shape -> [b, 3, 300, 300]
        conv4_3_feats, conv7_feats = self.base(x)  # [b, 512, 38, 38], [b, 1024, 19, 19]

        # 对conv4_3使用L2规范化
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # [b, 1, 38, 38]
        conv4_3_feats = conv4_3_feats / norm  # [b, 512, 38, 38]
        conv4_3_feats = conv4_3_feats * self.rescale_factors  # [b, 512, 38, 38]

        # [b, 512, 10, 10], [b, 256, 5, 5], [b, 256, 3, 3], [b, 256, 1, 1]
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = self.aux_convs(conv7_feats)

        # [b, 8732, 4], [b, 8732, n_classes]
        loc, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats,
                                              conv9_2_feats, conv10_2_feats, conv11_2_feats)

        return loc, classes_scores

    @staticmethod
    def create_prior_boxes():
        # 特征图的尺寸
        features_dim = {'conv4_3': 38, 'conv7': 19, 'conv8_2': 10,
                        'conv9_2': 5, 'conv10_2': 3, 'conv11_2': 1}
        # prior的scale
        object_scales = {'conv4_3': 0.1, 'conv7': 0.2, 'conv8_2': 0.375,
                         'conv9_2': 0.55, 'conv10_2': 0.725, 'conv11_2': 0.9}
        # prior的aspect ratio
        # conv7，conv8_2和conv9_2会多出 3:1 和 1:3
        aspect_ratios = {
            'conv4_3': [1., 2., 0.5],
            'conv7': [1., 2., 3., 0.5, 0.333],
            'conv8_2': [1., 2., 3., 0.5, 0.333],
            'conv9_2': [1., 2., 3., 0.5, 0.333],
            'conv10_2': [1., 2., 0.5],
            'conv11_2': [1., 2., 0.5]
        }
        # 记录特征图的名称，用来查找当前特征图的下一个特征图
        features_name = list(features_dim.keys())
        # 所有的priors
        prior_boxes = []
        # 每个特征图都会有priors
        for k, feature in enumerate(features_name):
            # 每个特征图的每个位置都有priors
            # 模仿卷积的操作，按照从左到右，从上到下的顺序计算priors，为了与预测卷积相匹配
            for i in range(features_dim[feature]):
                for j in range(features_dim[feature]):
                    # 当前特征图的当前格子的中心坐标（需要进行缩放）
                    cx = (j + 0.5) / features_dim[feature]
                    cy = (i + 0.5) / features_dim[feature]
                    # 为当前格子按照aspect ratios生成priors
                    for ratio in aspect_ratios[feature]:
                        # w = s * sqrt(a)， h = s / sqrt(a)
                        # 计算每个prior的中心坐标
                        prior_boxes.append([cx, cy, object_scales[feature] * sqrt(ratio),
                                            object_scales[feature] / sqrt(ratio)])
                        # 当ratio时，需要额外添加一个prior
                        if ratio == 1.:
                            # 如果当前特征图不是最后一个特征图，即当前特征图不是conv11_2
                            if k != len(features_name) - 1:
                                # 那么这个额外的prior的scale就是 sqrt(当前特征图scale * 下一个特征图scale)
                                additional_scale = sqrt(object_scales[feature] * object_scales[features_name[k + 1]])
                            else:
                                # 如果当前特征图是最后一个，它就不存在下一个特征图，直接将scale设置为1
                                additional_scale = 1.
                            # 添加额外的prior的中心坐标
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])
        # 最后将所有priors转换为一个tensor
        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # [8732, 4]
        return prior_boxes

    def detect_objects(self, predicted_loc, predicted_scores, min_score, max_overlap, top_k):
        """
        根据预测结果检测物体
        :param predicted_loc: 预测的偏移量 [b, 8732, 4]
        :param predicted_scores: 预测的分数 [b, 8732, n_classes]
        :param min_score:  类别最低分数，如果低于此分数则认为这个物体不是该类
        :param max_overlap: 最大重叠程度的阈值，非极大抑制所需
        :param top_k: 最终只保留前k个结果
        :return: 经过一系列筛选后的预测结果
        """
        batch_size = predicted_loc.size(0)
        n_priors = self.prior_cxcy_boxes.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # [b, 8732, n_classes]

        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_loc.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # 将对priors预测的偏移量转化为边界坐标
            decoded_loc = cxcy_to_xy(gcxcy_to_cxcy(predicted_loc[i], self.prior_cxcy_boxes))  # [8732, 4]

            image_boxes = list()
            image_labels = list()
            image_scores = list()

            # 检查每一个类别
            for c in range(1, self.n_classes):  # 类别从1开始，0表示背景类
                # 仅保留预测类别分数超过最低分数的预测框和类别
                class_scores = predicted_scores[i][:, c]  # [8732]
                score_above_min_score = class_scores > min_score  # torch.uint8 tensor, 索引
                n_above_min_score = score_above_min_score.sum().item()
                # 如果预测分数没有超过最低分的，则该图片认为不含物体
                if n_above_min_score == 0:
                    continue

                class_scores = class_scores[score_above_min_score]
                class_decoded_loc = decoded_loc[score_above_min_score]

                # 对预测框和类别，按照类别得分排序
                class_scores, sort_index = class_scores.sort(dim=0, descending=True)
                class_decoded_loc = class_decoded_loc[sort_index]

                # 查找预测框之间的重叠
                # 返回一个 [n, n] 的张量，表示每个预测框与其他所有预测框的IoU值
                overlap = find_jaccard_overlap(class_decoded_loc, class_decoded_loc)  # [n, n]

                # 非极大抑制
                # 记录要抑制的box，1表示抑制，0表示不抑制
                suppress = torch.zeros(n_above_min_score, dtype=torch.uint8).to(device)
                for box in range(class_decoded_loc.size(0)):
                    # 如果该box已经标记为抑制，则不必再次进行检测
                    if suppress[box] == 1:
                        continue
                    # 抑制重叠大于允许最大重叠的box
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # 自身与自身重叠为1，但是不应该抑制本身
                    suppress[box] = 0
                image_boxes.append(class_decoded_loc[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            if len(image_boxes) == 0:
                # 如果没有任何类别被检测到，则为背景存一个占位符
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.FloatTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # 拼接为单个tensor
            image_boxes = torch.cat(image_boxes, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            n_objects = image_scores.size(0)

            # 仅保留前k个对象
            if n_objects > top_k:
                image_scores, sort_index = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_index][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_index][:top_k]  # (top_k)

            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores
