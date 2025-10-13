import os
from motion_blur_generator import QRMotionBlurGenerator



def generate_comprehensive_dataset(qr_image_path: str, output_dir ='E:/project_deblur/dataset/blur/motion_claude/'):
    """
    生成完整的二维码运动模糊数据集

    :param qr_image_path: 二维码图像路径
    :param output_dir: 输出目录
    """
    # 创建生成器
    generator = QRMotionBlurGenerator(qr_image_path)

    # 定义参数
    motion_lengths = list(range(20, 61, 5))  # 0-15 pixels
    motion_angles = list(range(0, 181, 10))  # 0-180 degrees, interval 20
    motion_angles = motion_angles + [45, 135]

    print(f"Generating dataset with {len(motion_lengths)} lengths and {len(motion_angles)} angles...")
    print(f"Total images to generate: {len(motion_lengths) * len(motion_angles)}")

    # 生成数据集
    dataset_info = generator.generate_motion_blur_dataset(
        motion_lengths=motion_lengths,
        motion_angles=motion_angles,
        output_dir=output_dir,
        use_non_uniform=True
    )

    print(f"\nDataset generation complete!")
    print(f"Total images generated: {len(dataset_info)}")
    print(f"Output directory: {output_dir}")

    # 生成统计信息
    print("\nDataset Statistics:")
    print(f"Motion lengths: {min(motion_lengths)} - {max(motion_lengths)} pixels")
    print(f"Motion angles: {min(motion_angles)} - {max(motion_angles)} degrees")
    print(f"Angle interval: 20 degrees")

    return dataset_info


# 使用示例
if __name__ == "__main__":
    # 示例用法
    #qr_image_path = 'E:/project_deblur/dataset/blur/clear'  # 替换为您的二维码图像路径
    folder = 'E:/project_deblur/dataset/blur/clear'
    output_path = 'E:/project_deblur/dataset/blur/motion_claude/'
    for path in os.listdir(folder):
        qr_image_path = os.path.join(folder, path).replace('\\', '/')
        # 创建生成器
        generator = QRMotionBlurGenerator(qr_image_path)

        # 演示单个模糊效果
        #print("Generating demo blur...")
        #generator.demo_single_blur(length=8, angle=45, use_non_uniform=True)

        # 生成完整数据集
        print("\nGenerating complete dataset...")
        dataset_info = generate_comprehensive_dataset(qr_image_path, output_path)

        # 显示一些统计信息
        print(f"\nGenerated {len(dataset_info)} motion blur variants")
        print("Sample entries:")
        for i, info in enumerate(dataset_info[:5]):
            print(f"{i + 1}. {info['filename']} - Length: {info['motion_length']}px, Angle: {info['motion_angle']}°")

