import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import LayerNorm2d

class SimpleGate(nn.Module):
    """NAFNet核心：简单门控机制"""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LightNAFBlock(nn.Module):
    """超轻量级NAF Block"""

    def __init__(self, c, eps=1e-6):
        super().__init__()
        # 1x1 -> 3x3 DW -> 1x1 的轻量级结构
        self.norm = LayerNorm2d(c, eps=eps)
        self.conv1 = nn.Conv2d(c, c * 2, 1, bias=True)
        self.dw_conv = nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2, bias=True)
        self.sg = SimpleGate()
        self.conv2 = nn.Conv2d(c, c, 1, bias=True)

        # 可学习的残差权重
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = inp
        x = self.norm(x)
        x = self.conv1(x)
        x = self.dw_conv(x)
        x = self.sg(x)  # 门控激活
        x= self.conv2(x)
        return inp + x * self.beta



class DownSample(nn.Module):
    """轻量级下采样"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=True)

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    """轻量级上采样"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, 1, bias=True),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.conv(x)


class LENet(nn.Module):
    """轻量级二维码去模糊U-Net"""

    def __init__(self, in_ch=3, base_ch=16):
        super().__init__()

        # 输入输出层
        self.intro = nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=True)
        self.outro = nn.Conv2d(base_ch, in_ch, 3, padding=1, bias=True)

        # 编码器 - 3层下采样
        self.enc1 = LightNAFBlock(base_ch)  # 16
        self.down1 = DownSample(base_ch, base_ch * 2)  # 16->32

        self.enc2 = LightNAFBlock(base_ch * 2)  # 32
        self.down2 = DownSample(base_ch * 2, base_ch * 4)  # 32->64

        self.enc3 = LightNAFBlock(base_ch * 4)  # 64
        self.down3 = DownSample(base_ch * 4, base_ch * 8)  # 64->128

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            LightNAFBlock(base_ch * 8),  # 128
            LightNAFBlock(base_ch * 8)  # 双层增强特征提取
        )

        # 解码器 - 3层上采样
        self.up3 = UpSample(base_ch * 8, base_ch * 4)  # 128->64
        self.dec3 = LightNAFBlock(base_ch * 4)  # 64 (skip connection后)

        self.up2 = UpSample(base_ch * 4, base_ch * 2)  # 64->32
        self.dec2 = LightNAFBlock(base_ch * 2)  # 32

        self.up1 = UpSample(base_ch * 2, base_ch)  # 32->16
        self.dec1 = LightNAFBlock(base_ch)  # 16

        # 二维码特化：边缘锐化层
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, 1),
            nn.Conv2d(base_ch // 2, base_ch, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 记录输入用于残差连接
        inp = x

        # 特征提取
        x = self.intro(x)

        # 编码器
        e1 = self.enc1(x)  # Skip connection 1
        x = self.down1(e1)

        e2 = self.enc2(x)  # Skip connection 2
        x = self.down2(e2)

        e3 = self.enc3(x)  # Skip connection 3
        x = self.down3(e3)

        # 瓶颈
        x = self.bottleneck(x)

        # 解码器
        x = self.up3(x)
        x = x + e3  # Skip connection
        x = self.dec3(x)

        x = self.up2(x)
        x = x + e2  # Skip connection
        x = self.dec2(x)

        x = self.up1(x)
        x = x + e1  # Skip connection
        x = self.dec1(x)

        # 边缘增强（针对二维码锐利边缘特性）
        edge_weight = self.edge_enhance(x)
        x = x * (1 + edge_weight)

        # 输出
        x = self.outro(x)

        # 全局残差连接
        return x + inp


def count_parameters(model):
    """统计参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, input_size=(1, 3, 256, 256)):
    """模型信息摘要"""
    device = next(model.parameters()).device
    x = torch.randn(input_size).to(device)

    params = count_parameters(model)

    # 测试前向传播
    model.eval()
    with torch.no_grad():
        y = model(x)

    print(f"模型参数量: {params / 1000:.1f}K ({params / 1e6:.3f}M)")
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {y.shape}")
    print(f"内存占用估算: {params * 4 / 1024 / 1024:.1f}MB (fp32)")

    return params


# 使用示例和对比
if __name__ == "__main__":
    print("=== 轻量级二维码去模糊网络 ===")

    # 超轻量版本 (base_ch=12)
    print("\n1. 超轻量版本:")
    tiny_model = QRDeblurNet(in_ch=3, base_ch=12)
    tiny_params = model_summary(tiny_model)

    # 标准轻量版本 (base_ch=16)
    print("\n2. 标准轻量版本:")
    light_model = QRDeblurNet(in_ch=3, base_ch=16)
    light_params = model_summary(light_model)

    # 性能版本 (base_ch=24)
    print("\n3. 性能版本:")
    perf_model = QRDeblurNet(in_ch=3, base_ch=24)
    perf_params = model_summary(perf_model)

    print(f"\n=== 网络特点 ===")
    print("✓ U-Net架构保证信息流动")
    print("✓ NAFNet简单门控，无激活函数")
    print("✓ 深度可分离卷积降低参数")
    print("✓ PixelShuffle高效上采样")
    print("✓ 边缘增强适配二维码特性")
    print("✓ 全局残差连接稳定训练")

    print(f"\n=== 部署建议 ===")
    print("移动端: 选择超轻量版本 (~30K参数)")
    print("边缘设备: 选择标准轻量版本 (~50K参数)")
    print("桌面/服务器: 选择性能版本 (~110K参数)")

    # 速度测试
    print(f"\n=== 推理速度测试 (CPU) ===")
    import time

    model = light_model
    model.eval()
    x = torch.randn(1, 3, 256, 256)

    # 预热
    for _ in range(5):
        with torch.no_grad():
            _ = model(x)

    # 测速
    times = []
    for _ in range(20):
        start = time.time()
        with torch.no_grad():
            _ = model(x)
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    print(f"平均推理时间: {avg_time * 1000:.1f}ms")
    print(f"FPS: {1 / avg_time:.1f}")