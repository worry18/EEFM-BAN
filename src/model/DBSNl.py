import torch
import torch.nn as nn
from torchvision import transforms

from . import regist_model
import matplotlib.pyplot as plt
from TMFE import OVH_MaskedConv2d, OVHMaskedConv2d, SquareMaskedConv2d


@regist_model
class DBSNl(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=128, num_module=12):
        super().__init__()

        assert base_ch % 2 == 0, "base channel should be divided with 2"
        ly = []
        # 浅层特征提取
        ly += [nn.Conv2d(in_ch, base_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        self.head = nn.Sequential(*ly)

        # 深层特征提取
        self.branch1 = DC_branchl(2, base_ch, num_module)
        self.branch2 = DC_branchl(3, base_ch, num_module)
        self.branch3 = DC_branchl3(3, base_ch, num_module)
        self.sigmoid = nn.Sigmoid()

        # 特征融合
        ly = []
        ly += [nn.Conv2d(base_ch * 2, base_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, out_ch, kernel_size=1)]
        self.tail = nn.Sequential(*ly)

    def forward(self, x):
        x = self.head(x)
        br1 = self.branch1(x)
        br2 = self.branch2(x)
        br3 = self.branch3(x)
        # x = torch.cat([br1, br2, br3], dim=1)
        # x = x * 2 * self.sigmoid(x)
        # x = self.tail(x)

        # x1 = torch.cat([br1, br2], dim=1)
        # x2 = torch.cat([br2, br3], dim=1)
        # x = x1 + x2
        # x = self.tail(x)

        # x1 = torch.cat([br1, br2], dim=1)
        # x2 = torch.cat([br1, br3], dim=1)
        # x3 = torch.cat([br2, br3], dim=1)
        # x = x1 + x2 + x3
        # x = x * 2 * self.sigmoid(x)
        # x = self.tail(x)

        x1 = torch.cat([br1, br2], dim=1)
        x2 = torch.cat([br1, br3], dim=1)
        x = x1 + x2
        x = x * 2 * self.sigmoid(x)
        x = self.tail(x)
        return x


    def _initialize_weights(self):
        # Liyong version
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)


class DC_branchl(nn.Module):
    def __init__(self, stride, in_ch, num_module):
        super().__init__()

        ly = []
        if stride == 2:
            ly += [SquareMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        elif stride == 3:
            ly += [OVHMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]

        if stride == 2:
            self.ddc_layers = [MPIE(stride, in_ch) for _ in range(num_module)]
        elif stride == 3:
            self.ddc_layers = [DCl(stride, in_ch) for _ in range(num_module-3)]

        ly += self.ddc_layers
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]

        self.body = nn.Sequential(*ly)

    def forward(self, x):
        out = x
        for i, ddc_layer in enumerate(self.body):
            out = ddc_layer(out)
        return out


class DC_branchl3(nn.Module):
    def __init__(self, stride, in_ch, num_module):
        super().__init__()

        ly = []
        ly += [OVH_MaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [MPIE(stride, in_ch) for _ in range(num_module)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]

        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return self.body(x)


class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)