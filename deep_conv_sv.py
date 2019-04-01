import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, bias=False, **kwargs)
        self.bn2d = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn2d(x)
        x = self.prelu(x)
        return x

class Stem(nn.Module):
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        kwargs1 = {'kernel_size': (3, 1), 'padding': 0, 'stride': 1}
        self.conv1 = BasicConv2d(in_channels, 32, **kwargs1)
        kwargs2 = {'kernel_size': (3, 3), 'padding': 0, 'stride': 1}
        self.conv2 = BasicConv2d(32, 32, **kwargs2)
        kwargs3 = {'kernel_size': (3, 3), 'padding': 1, 'stride': 1}
        self.conv3 = BasicConv2d(32, 64, **kwargs3)
        kwargs4 = {'kernel_size': (1, 1), 'padding': 0, 'stride': 1}
        self.conv4 = BasicConv2d(64, 80, **kwargs4)
        kwargs5 = {'kernel_size': (3, 3), 'padding': 0, 'stride': 1, }
        self.conv5 = BasicConv2d(80, 192, **kwargs5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, kernel_size=(3, 3), stride=2)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class Inception_resnet_A(nn.Module):
    def __init__(self):
        super(Inception_resnet_A, self).__init__()
        self.conv2d_1 = BasicConv2d(192, 32, kernel_size = 1)
        self.conv2d_2_1 = BasicConv2d(192, 32, kernel_size = 1)
        self.conv2d_2_2 = BasicConv2d(32, 32, kernel_size = 3, padding = 1)
        self.conv2d_3_1 = BasicConv2d(192, 32, kernel_size = 1)
        self.conv2d_3_2 = BasicConv2d(32, 32, kernel_size = 3, padding = 1)
        self.conv2d_3_3 = BasicConv2d(32, 32, kernel_size = 3, padding = 1)
        self.conv2d_final = nn.Conv2d(96, 192, kernel_size = 1)
        self.bn_final = nn.BatchNorm2d(192)

    def forward(self, x):
        out1 = self.conv2d_1(x)

        out2 = self.conv2d_2_1(x)
        out2 = self.conv2d_2_2(out2)

        out3 = self.conv2d_3_1(x)
        out3 = self.conv2d_3_2(out3)
        out3 = self.conv2d_3_3(out3)

        out = torch.cat((out1, out2, out3), dim = 1)
        out = self.conv2d_final(out)
        out = out + x
        out = self.bn_final(out)
        out = F.relu(out, inplace = True)
        return out

class Reduction_A(nn.Module):
    def __init__(self):
        super(Reduction_A, self).__init__()
        self.conv2d_1 = BasicConv2d(192, 384, kernel_size = (3,3), stride = 2)
        self.conv2d_2_1 = BasicConv2d(192, 192, kernel_size = (1,1))
        self.conv2d_2_2 = BasicConv2d(192, 192, kernel_size = (3,3), padding = 1)
        self.conv2d_2_3 = BasicConv2d(192, 256, kernel_size = (3,3), stride = 2)

    def forward(self, x):
        out1 = self.conv2d_1(x)
        out2 = self.conv2d_2_1(x)
        out2 = self.conv2d_2_2(out2)
        out2 = self.conv2d_2_3(out2)
        out3 = F.max_pool2d(x, kernel_size = (3,3), stride = 2)
        out = torch.cat((out1, out2, out3), dim = 1)
        return out

class Inception_resnet_B(nn.Module):
    def __init__(self):
        super(Inception_resnet_B, self).__init__()
        self.conv2d_1 = BasicConv2d(832, 128, kernel_size = 1)
        self.conv2d_2_1 = BasicConv2d(832, 128, kernel_size = 1)
        self.conv2d_2_2 = BasicConv2d(128, 128, kernel_size = (1,7), padding = (0, 3))
        self.conv2d_2_3 = BasicConv2d(128, 128, kernel_size = (7,1), padding = (3, 0))
        self.conv2d_final = nn.Conv2d(256, 832, kernel_size = 1)
        self.bn_final = nn.BatchNorm2d(832)

    def forward(self, x):
        out1 = self.conv2d_1(x)
        
        out2 = self.conv2d_2_1(x)
        out2 = self.conv2d_2_2(out2)
        out2 = self.conv2d_2_3(out2)

        out = torch.cat((out1, out2), dim = 1)
        out = self.conv2d_final(out)
        out = out + x
        out = self.bn_final(out)
        out = F.relu(out, inplace=True)
        return out

class Reduction_B(nn.Module):
    def __init__(self):
        super(Reduction_B, self).__init__()
        self.conv2d_1_1 = BasicConv2d(832, 256, kernel_size = 1)
        self.conv2d_1_2 = BasicConv2d(256, 384, kernel_size = 3, stride = 2)
        self.conv2d_2_1 = BasicConv2d(832, 256, kernel_size = 1)
        self.conv2d_2_2 = BasicConv2d(256, 256, kernel_size = 3, stride = 2)
        self.conv2d_3_1 = BasicConv2d(832, 256, kernel_size = 1)
        self.conv2d_3_2 = BasicConv2d(256, 256, kernel_size = 3, padding = 1)
        self.conv2d_3_3 = BasicConv2d(256, 256, kernel_size = 3, stride = 2)

    def forward(self, x):
        out1 = self.conv2d_1_1(x)
        out1 = self.conv2d_1_2(out1)
        out2 = self.conv2d_2_1(x)
        out2 = self.conv2d_2_2(out2)
        out3 = self.conv2d_3_1(x)
        out3 = self.conv2d_3_2(out3)
        out3 = self.conv2d_3_3(out3)
        out4 = F.max_pool2d(x, kernel_size = (3,3), stride = 2)
        out = torch.cat((out4,out1,out2,out3), dim = 1)
        return out

class Inception_resnet_C(nn.Module):
    def __init__(self):
        super(Inception_resnet_C, self).__init__()
        self.conv2d_1 = BasicConv2d(1728, 192, kernel_size = 1)
        self.conv2d_2_1 = BasicConv2d(1728, 192, kernel_size = 1)
        self.conv2d_2_2 = BasicConv2d(192, 192, kernel_size = (1,3), padding = (0,1))
        self.conv2d_2_3 = BasicConv2d(192, 192, kernel_size = (3,1), padding = (1,0))
        self.conv2d_final = nn.Conv2d(384, 1728, kernel_size = 1)
        self.bn_final = nn.BatchNorm2d(1728)

    def forward(self, x):
        out1 = self.conv2d_1(x)

        out2 = self.conv2d_2_1(x)
        out2 = self.conv2d_2_2(out2)
        out2 = self.conv2d_2_3(out2)

        out = torch.cat((out1, out2), dim = 1)
        out = self.conv2d_final(out)
        out = out + x
        out = self.bn_final(out)
        out = F.relu(out, inplace = True)
        return out

class SpeakerVerification(nn.Module):
    def __init__(self, num_classes, a = 5, b = 10, c = 5):
        super(SpeakerVerification, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.b = b
        self.c = c
        self.stem = Stem(1)
        self.inception_a = self._make_layers(self.a, Inception_resnet_A)
        self.reduction_a = Reduction_A()
        self.inception_b = self._make_layers(self.b, Inception_resnet_B)
        self.reduction_b = Reduction_B()
        self.inception_c = self._make_layers(self.c, Inception_resnet_C)
        self.pool = nn.AdaptiveAvgPool2d([1,1])
        self.fc1 = nn.Linear(1728, 512)
        self.bn = nn.BatchNorm1d(512)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, self.num_classes)

    def _make_layers(self, n, Inception_type):
        layers = []
        for _ in range(n):
            layers.append(Inception_type())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.inception_a(out)
        out = self.reduction_a(out)
        out = self.inception_b(out)
        out = self.reduction_b(out)
        out = self.inception_c(out)
        out = self.pool(out)
        out = out.view(-1, 1728)
        embedding = self.fc1(out)
        out = self.bn(embedding)
        out = self.prelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return embedding, out