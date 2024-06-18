import torch
import torch.nn as nn

class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        return x

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * BottleNeck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN_layer = BasicCNN()

        # Define the layers
        self.layer1 = self.make_layer(64, 64, 3)
        self.layer2 = self.make_layer(256, 128, 4, stride=2)
        self.layer3 = self.make_layer(512, 256, 6, stride=2)
        self.layer4 = self.make_layer(1024, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BottleNeck.expansion, 1000)

    def make_layer(self, in_channels, out_channels, blocks_number, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            )

        layers = []
        layers.append(BottleNeck(in_channels, out_channels, stride, downsample))
        in_channels = out_channels * BottleNeck.expansion
        for _ in range(1, blocks_number):
            layers.append(BottleNeck(in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.CNN_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# model = ResNet50()
# input_tensor = torch.randn(1, 3, 224, 224)
# output = model(input_tensor)
# print(output.shape)  # 应该是 (1, 1000)


def make_layer(in_channels, out_channels, blocks_number, stride=1):
    downsample = None
    if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

    layers = []
    layers.append(BottleNeck(in_channels, out_channels, stride, downsample))
    print(in_channels,out_channels)
    in_channels = out_channels * BottleNeck.expansion
    print(in_channels,out_channels)
    for _ in range(1, blocks_number):
        layers.append(BottleNeck(in_channels, out_channels))
        print(in_channels,out_channels)
    return nn.Sequential(*layers)

print(make_layer(64,256,3))