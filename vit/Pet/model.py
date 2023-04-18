import torch
import torchvision

net = torchvision.models.vit_b_32(pretrained=True, num_classes=37,
                                  depth=4,dropout=0.3)

