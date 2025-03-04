import torch
import torch.nn as nn


class SquareMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, 1:-1, 1:-1] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class FMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(0)
        self.mask[:, :, 0, 1:kH - 1] = 1
        self.mask[:, :, -1, 1:kH - 1] = 1
        self.mask[:, :, :, kH // 2] = 0
        self.mask[:, :, kW // 2, :] = 1
        self.mask[:, :, kW // 2, kH // 2] = 0
        self.mask[:, :, kW // 2 - 1, kH // 2] = 0
        self.mask[:, :, kW // 2 + 1, kH // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class OHHVMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(0)
        self.mask[:, :, :, kH // 2] = 1
        self.mask[:, :, 1:kH - 1, 0] = 1
        self.mask[:, :, kW // 2, :] = 1
        self.mask[:, :, kW // 2, kH // 2] = 0
        self.mask[:, :, 1:kH - 1, -1] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class F_MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, 0, 1:kH - 1] = 0
        self.mask[:, :, -1, 1:kH - 1] = 0
        self.mask[:, :, :, kH // 2] = 1
        self.mask[:, :, kW // 2, :] = 0
        self.mask[:, :, kW // 2, kH // 2] = 0
        self.mask[:, :, kW // 2 - 1, kH // 2] = 1
        self.mask[:, :, kW // 2 + 1, kH // 2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class OHHV_MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, :, kH // 2] = 0
        self.mask[:, :, kW // 2, :] = 0
        self.mask[:, :, kW // 2, kH // 2] = 0
        self.mask[:, :, kW // 2 - 1, 0] = 0
        self.mask[:, :, kW // 2 + 1, 0] = 0
        self.mask[:, :, kW // 2 + 1, kH - 1] = 0
        self.mask[:, :, kW // 2 - 1, kH - 1] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


# 可视化特定通道的特征图
# def visualize_feature_map(feature_map, idx):
#     feature_map = feature_map[0, idx].cpu().detach().numpy()  # 取第一个样本的第 idx 通道
#     plt.imshow(feature_map, cmap='viridis')
#     plt.colorbar()
#     plt.title(f"Feature Map Channel {idx}")
#     plt.show()


class OVVHMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(0)
        self.mask[:, :, :, kH // 2] = 1
        self.mask[:, :, 1:kH - 1, 0] = 1
        self.mask[:, :, 1:kH - 1, -1] = 1
        self.mask[:, :, kW // 2, :] = 0
        self.mask[:, :, kW // 2, kH // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class OVVH_MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, :, kH // 2] = 0
        self.mask[:, :, 1:kH - 1, 0] = 0
        self.mask[:, :, 1:kH - 1, -1] = 0
        self.mask[:, :, kW // 2, :] = 1
        self.mask[:, :, kW // 2, kH // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class OVMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        for i in range(kH):
            self.mask[:, :, i, i] = 0
        for i in range(kH):
            self.mask[:, :, kW - 1 - i, i] = 0
        self.mask[:, :, :, kH // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class OV_MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(0)
        for i in range(kH):
            self.mask[:, :, i, i] = 1
        for i in range(kH):
            self.mask[:, :, kW - 1 - i, i] = 1
        self.mask[:, :, :, kH // 2] = 1
        self.mask[:, :, kW // 2, kH // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class OVHMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        for i in range(kH):
            self.mask[:, :, i, i] = 0
        for i in range(kH):
            self.mask[:, :, kW - 1 - i, i] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class OVH_MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(0)
        for i in range(kH):
            self.mask[:, :, i, i] = 1
        for i in range(kH):
            self.mask[:, :, kW - 1 - i, i] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class OHMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        for i in range(kH):
            self.mask[:, :, i, i] = 0
        for i in range(kH):
            self.mask[:, :, kW - 1 - i, i] = 0
        self.mask[:, :, kW // 2, :] = 0
        self.mask[:, :, kW // 2, kH // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class OH_MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(0)
        for i in range(kH):
            self.mask[:, :, i, i] = 1
        for i in range(kH):
            self.mask[:, :, kW - 1 - i, i] = 1
        self.mask[:, :, kW // 2, :] = 1
        self.mask[:, :, kW // 2, kH // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)