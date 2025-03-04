import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# 转换torch张量为numpy数组，并调整图像通道顺序从RGB到BGR
def tensor2np(t: torch.Tensor):
    t = t.cpu().detach()
    if len(t.shape) == 2:  # 灰度图
        return t.permute(1, 2, 0).numpy()
    else:  # 彩色图像
        return np.flip(t.permute(1, 2, 0).numpy(), axis=2)  # RGB转BGR


# 计算PSNR
def psnr(img1, img2):
    if isinstance(img1, torch.Tensor):
        img1 = tensor2np(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor2np(img2)

    img1 = np.clip(img1, 0, 255)
    img2 = np.clip(img2, 0, 255)
    return peak_signal_noise_ratio(img1, img2, data_range=255)


# 计算SSIM
def ssim(img1, img2):
    if isinstance(img1, torch.Tensor):
        img1 = tensor2np(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor2np(img2)

    img1 = np.clip(img1, 0, 255)
    img2 = np.clip(img2, 0, 255)
    return structural_similarity(img1, img2, multichannel=True, data_range=255, channel_axis=2)


# 用Pillow打开图像   311  47  560  578  745
img1 = Image.open("/home/wry/AP-BSN-master-ICONIP/0085_DN_33.18_0.884_0.146_0.116.png")
img2 = Image.open("/home/wry/AP-BSN-master-ICONIP/0085_CL.png")

# 转换Pillow图像为numpy数组
img1_np = np.array(img1)
img2_np = np.array(img2)

# 确保图像是三个通道RGB格式
if img1_np.shape[-1] == 4 or img2_np.shape[-1] == 4:
    img1_np = img1_np[..., :3]  # 删除透明的alpha通道
    img2_np = img2_np[..., :3]

# 将numpy数组转化为torch张量
img1_tensor = torch.from_numpy(img1_np).permute(2, 0, 1).float()  # (h, w, c) -> (c, h, w)
img2_tensor = torch.from_numpy(img2_np).permute(2, 0, 1).float()

# 计算PSNR和SSIM值
psnr_value = psnr(img1_tensor, img2_tensor)
ssim_value = ssim(img1_tensor, img2_tensor)

# 打印结果
print(f"PSNR value: {psnr_value:.4f}")
print(f"SSIM value: {ssim_value:.4f}")

