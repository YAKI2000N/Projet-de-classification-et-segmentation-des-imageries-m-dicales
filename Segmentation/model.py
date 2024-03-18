import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv9 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(1024)
        self.conv10 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(1024)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv13 = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(256)
        self.conv14 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv15 = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
        self.bn15 = nn.BatchNorm2d(128)
        self.conv16 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn16 = nn.BatchNorm2d(128)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv17 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.bn17 = nn.BatchNorm2d(64)
        self.conv18 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn18 = nn.BatchNorm2d(64)

        self.conv19 = nn.Conv2d(64, n_classes, kernel_size=1) 

        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1p = self.pool1(x1)

        x2 = F.relu(self.bn3(self.conv3(x1p)))
        x2 = F.relu(self.bn4(self.conv4(x2)))
        x2p = self.pool2(x2)

        x3 = F.relu(self.bn5(self.conv5(x2p)))
        x3 = F.relu(self.bn6(self.conv6(x3)))
        x3p = self.pool3(x3)

        x4 = F.relu(self.bn7(self.conv7(x3p)))
        x4 = F.relu(self.bn8(self.conv8(x4)))
        x4p = self.pool4(x4)

        x5 = F.relu(self.bn9(self.conv9(x4p)))
        x5 = F.relu(self.bn10(self.conv10(x5)))

        # Decoder
        x5u = self.upconv1(x5)
        x4cat = torch.cat((x5u, x4), dim=1)
        x4u = F.relu(self.bn11(self.conv11(x4cat)))
        x4u = F.relu(self.bn12(self.conv12(x4u)))

        x4uu = self.upconv2(x4u)
        x3cat = torch.cat((x4uu, x3), dim=1)
        x3u = F.relu(self.bn13(self.conv13(x3cat)))
        x3u = F.relu(self.bn14(self.conv14(x3u)))

        x3uu = self.upconv3(x3u)
        x2cat = torch.cat((x3uu, x2), dim=1)
        x2u = F.relu(self.bn15(self.conv15(x2cat)))
        x2u = F.relu(self.bn16(self.conv16(x2u)))

        x2uu = self.upconv4(x2u)
        x1cat = torch.cat((x2uu, x1), dim=1)
        x1u = F.relu(self.bn17(self.conv17(x1cat)))
        x1u = F.relu(self.bn18(self.conv18(x1u)))

        x = self.conv19(x1u)
        return x
