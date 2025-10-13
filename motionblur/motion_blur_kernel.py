import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image
import os
import math
from typing import List, Tuple, Optional


class MotionBlurKernel:
    """
    生成运动模糊核的类
    """

    def __init__(self, length: float, angle: float, kernel_size: Optional[int] = None):
        """
        初始化运动模糊核

        :param length: 运动长度 (0-15 pixels)
        :param angle: 运动角度 (0-180 degrees)
        :param kernel_size: 核大小，如果为None则自动计算
        """
        self.length = length
        self.angle = angle
        self.kernel_size = kernel_size
        self.kernel = None

    def generate_linear_kernel(self) -> np.ndarray:
        """
        生成线性运动模糊核
        """
        if self.length == 0:
            return np.array([[1]])

        # 自动计算核大小
        if self.kernel_size is None:
            self.kernel_size = int(2 * self.length) + 1

        # 确保核大小为奇数
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        kernel = np.zeros((self.kernel_size, self.kernel_size))
        center = self.kernel_size // 2

        # 将角度转换为弧度
        angle_rad = np.deg2rad(self.angle)

        # 计算运动方向
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)

        # 生成运动轨迹点
        points = []
        for i in range(int(self.length) + 1):
            t = i / max(1, self.length)
            x = center + t * self.length * dx - self.length * dx / 2
            y = center + t * self.length * dy - self.length * dy / 2
            points.append((x, y))

        # 使用Bresenham算法或插值填充运动轨迹
        self._fill_motion_path(kernel, points)

        # 归一化核
        if kernel.sum() > 0:
            kernel = kernel / kernel.sum()
        else:
            kernel[center, center] = 1.0

        self.kernel = kernel
        return kernel

    def generate_non_uniform_kernel(self, acceleration: float = 0.5) -> np.ndarray:
        """
        生成非均匀运动模糊核（模拟加速/减速运动）

        :param acceleration: 加速度参数，0.5为匀速，>0.5为加速，<0.5为减速
        """
        if self.length == 0:
            return np.array([[1]])

        if self.kernel_size is None:
            self.kernel_size = int(2 * self.length) + 1

        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        kernel = np.zeros((self.kernel_size, self.kernel_size))
        center = self.kernel_size // 2

        angle_rad = np.deg2rad(self.angle)
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)

        # 生成非均匀运动轨迹
        points = []
        weights = []

        num_points = int(self.length * 2) + 1
        for i in range(num_points):
            # 非线性时间参数
            t_linear = i / max(1, num_points - 1)
            t_non_linear = t_linear ** acceleration

            x = center + t_non_linear * self.length * dx - self.length * dx / 2
            y = center + t_non_linear * self.length * dy - self.length * dy / 2

            # 计算权重（速度相关）
            if i == 0:
                weight = 1.0
            else:
                prev_t = ((i - 1) / max(1, num_points - 1)) ** acceleration
                weight = 1.0 / max(0.1, t_non_linear - prev_t)

            points.append((x, y))
            weights.append(weight)

        # 填充带权重的运动轨迹
        self._fill_weighted_motion_path(kernel, points, weights)

        # 归一化
        if kernel.sum() > 0:
            kernel = kernel / kernel.sum()
        else:
            kernel[center, center] = 1.0

        self.kernel = kernel
        return kernel

    def _fill_motion_path(self, kernel: np.ndarray, points: List[Tuple[float, float]]):
        """
        填充运动路径到核中
        """
        for x, y in points:
            # 使用双线性插值
            x_int = int(x)
            y_int = int(y)
            x_frac = x - x_int
            y_frac = y - y_int

            # 检查边界
            if 0 <= x_int < kernel.shape[1] and 0 <= y_int < kernel.shape[0]:
                kernel[y_int, x_int] += (1 - x_frac) * (1 - y_frac)

            if 0 <= x_int + 1 < kernel.shape[1] and 0 <= y_int < kernel.shape[0]:
                kernel[y_int, x_int + 1] += x_frac * (1 - y_frac)

            if 0 <= x_int < kernel.shape[1] and 0 <= y_int + 1 < kernel.shape[0]:
                kernel[y_int + 1, x_int] += (1 - x_frac) * y_frac

            if 0 <= x_int + 1 < kernel.shape[1] and 0 <= y_int + 1 < kernel.shape[0]:
                kernel[y_int + 1, x_int + 1] += x_frac * y_frac

    def _fill_weighted_motion_path(self, kernel: np.ndarray, points: List[Tuple[float, float]], weights: List[float]):
        """
        填充带权重的运动路径
        """
        for (x, y), weight in zip(points, weights):
            x_int = int(x)
            y_int = int(y)
            x_frac = x - x_int
            y_frac = y - y_int

            if 0 <= x_int < kernel.shape[1] and 0 <= y_int < kernel.shape[0]:
                kernel[y_int, x_int] += weight * (1 - x_frac) * (1 - y_frac)

            if 0 <= x_int + 1 < kernel.shape[1] and 0 <= y_int < kernel.shape[0]:
                kernel[y_int, x_int + 1] += weight * x_frac * (1 - y_frac)

            if 0 <= x_int < kernel.shape[1] and 0 <= y_int + 1 < kernel.shape[0]:
                kernel[y_int + 1, x_int] += weight * (1 - x_frac) * y_frac

            if 0 <= x_int + 1 < kernel.shape[1] and 0 <= y_int + 1 < kernel.shape[0]:
                kernel[y_int + 1, x_int + 1] += weight * x_frac * y_frac

    def visualize_kernel(self, title: str = "Motion Blur Kernel"):
        """
        可视化运动模糊核
        """
        if self.kernel is None:
            print("Please generate kernel first!")
            return

        plt.figure(figsize=(8, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(self.kernel, cmap='hot', interpolation='nearest')
        plt.title(f'{title}\nLength: {self.length:.1f}px, Angle: {self.angle}°')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(self.kernel, cmap='gray', interpolation='nearest')
        plt.title('Grayscale View')
        plt.colorbar()

        plt.tight_layout()
        plt.show()