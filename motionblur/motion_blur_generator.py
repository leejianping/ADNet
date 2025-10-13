import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image
import os
import math
from typing import List, Tuple, Optional
from motion_blur_kernel import MotionBlurKernel

class QRMotionBlurGenerator:
    """
    专门用于二维码的运动模糊生成器
    """

    def __init__(self, qr_image_path: str):
        """
        初始化二维码运动模糊生成器

        :param qr_image_path: 二维码图像路径
        """
        self.qr_image_path = qr_image_path
        self.original_image = None
        self.load_image()

    def load_image(self):
        """
        加载二维码图像
        """
        if not os.path.exists(self.qr_image_path):
            raise FileNotFoundError(f"Image not found: {self.qr_image_path}")

        # 使用PIL加载图像
        pil_image = Image.open(self.qr_image_path)

        # 转换为RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        self.original_image = np.array(pil_image)

    def generate_motion_blur_dataset(self,
                                     motion_lengths: List[float] = None,
                                     motion_angles: List[float] = None,
                                     output_dir= 'E:/project_deblur/dataset/blur/motion_claude/',
                                     use_non_uniform: bool = True) -> List[dict]:
        """
        生成运动模糊数据集

        :param motion_lengths: 运动长度列表 (0-15 pixels)
        :param motion_angles: 运动角度列表 (0-180 degrees, interval 20)
        :param output_dir: 输出目录
        :param use_non_uniform: 是否使用非均匀模糊
        :return: 生成的数据集信息
        """
        if motion_lengths is None:
            motion_lengths = list(range(0, 16))  # 0-15 pixels

        if motion_angles is None:
            motion_angles = list(range(0, 181, 20))  # 0-180 degrees, interval 20

        # 创建输出目录
        #os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "kernels"), exist_ok=True)

        dataset_info = []

        for length in motion_lengths:
            for angle in motion_angles:
                # 生成运动模糊核
                kernel_generator = MotionBlurKernel(length, angle)

                if use_non_uniform:
                    # 随机选择加速度参数
                    acceleration = np.random.uniform(0.3, 0.7)
                    kernel = kernel_generator.generate_non_uniform_kernel(acceleration)
                else:
                    kernel = kernel_generator.generate_linear_kernel()

                # 应用运动模糊
                blurred_image = self.apply_motion_blur(kernel)

                # 保存图像
                ##提取图像名前缀
                imagename_with_ext = os.path.basename(self.qr_image_path)
                image_name = os.path.splitext(imagename_with_ext)[0]

                filename = f"{image_name}_l{length:02d}_a{angle:03d}.png"
                kernel_filename = f"kernel_l{length:02d}_a{angle:03d}.png"

                # 保存模糊图像
                cv2.imwrite(os.path.join(output_dir, "images", filename),
                            cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR))

                # 保存核
                kernel_vis = (kernel * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, "kernels", kernel_filename), kernel_vis)

                # 记录信息
                info = {
                    'filename': filename,
                    'kernel_filename': kernel_filename,
                    'motion_length': length,
                    'motion_angle': angle,
                    'acceleration': acceleration if use_non_uniform else 0.5,
                    'kernel_shape': kernel.shape,
                    'kernel_sum': kernel.sum()
                }
                dataset_info.append(info)

                print(f"Generated: {filename} (Length: {length}px, Angle: {angle}°)")

        # 保存数据集信息
        import json
        with open(os.path.join(output_dir, "dataset_info.json"), 'w') as f:
            json.dump(dataset_info, f, indent=2)

        return dataset_info

    def apply_motion_blur(self, kernel: np.ndarray) -> np.ndarray:
        """
        应用运动模糊到图像

        :param kernel: 运动模糊核
        :return: 模糊后的图像
        """
        if self.original_image is None:
            raise ValueError("No image loaded!")

        # 将图像转换为float类型
        image_float = self.original_image.astype(np.float32) / 255.0
        blurred_image = np.zeros_like(image_float)

        # 对每个颜色通道应用卷积
        for channel in range(image_float.shape[2]):
            blurred_image[:, :, channel] = signal.convolve2d(
                image_float[:, :, channel],
                kernel,
                mode='same',
                boundary='symm'
            )

        # 裁剪到有效范围并转换回uint8
        blurred_image = np.clip(blurred_image, 0, 1)
        blurred_image = (blurred_image * 255).astype(np.uint8)

        return blurred_image

    def demo_single_blur(self, length: float, angle: float, use_non_uniform: bool = True):
        """
        演示单个运动模糊效果

        :param length: 运动长度
        :param angle: 运动角度
        :param use_non_uniform: 是否使用非均匀模糊
        """
        # 生成核
        kernel_generator = MotionBlurKernel(length, angle)

        if use_non_uniform:
            kernel = kernel_generator.generate_non_uniform_kernel()
        else:
            kernel = kernel_generator.generate_linear_kernel()

        # 应用模糊
        blurred_image = self.apply_motion_blur(kernel)

        # 显示结果
        plt.figure(figsize=(15, 5))

        # 原图
        plt.subplot(1, 3, 1)
        plt.imshow(self.original_image)
        plt.title('Original QR Code')
        plt.axis('off')

        # 模糊核
        plt.subplot(1, 3, 2)
        plt.imshow(kernel, cmap='hot', interpolation='nearest')
        plt.title(f'Motion Kernel\nLength: {length}px, Angle: {angle}°')
        plt.colorbar()

        # 模糊结果
        plt.subplot(1, 3, 3)
        plt.imshow(blurred_image)
        plt.title('Blurred QR Code')
        plt.axis('off')

        plt.tight_layout()
        plt.show()