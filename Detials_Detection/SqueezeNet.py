import torch
import torch.nn as nn

class Fire(nn.Module):
    def __init__(self,inplanes,squeeze_planes,expand_planes):
        super(Fire,self).__init__()
        # Squeeze layer
        self.conv1 = nn.Conv2d(inplanes,squeeze_planes,kernel_size=1,stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ReLU(inplace=True)
        # Expande layer
        self.conv2 = nn.Conv2d(squeeze_planes,expand_planes,kernel_size=1,stride=1)
        self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(squeeze_planes,expand_planes,kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        # Concat
        out = torch.cat([out1,out2],1)
        out = self.relu2(out)
        return out
    

fire_block = Fire(512,128,512).cuda()
print(fire_block)



input = torch.randn(1,512,28,28).cuda() 
output = fire_block(input)
print(output.shape) # torch.Size([1, 1024, 28, 28])

