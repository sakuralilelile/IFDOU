import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 中文显示设置
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置Matplotlib后端
matplotlib.use('Agg')  # 非交互式后端，用于保存图片


# matplotlib.use('TkAgg')  # 交互式后端，用于显示窗口（取消注释此行可切换到交互式）


def plot_win_rate_and_adp(landlord_csv, farmer_csv):
    """
    读取CSV文件并分别绘制地主和农民的胜率及ADP曲线图

    参数:
    landlord_csv (str): 地主数据CSV文件的路径
    farmer_csv (str): 农民数据CSV文件的路径
    """
    try:
        # 读取地主CSV文件
        landlord_df = pd.read_csv(landlord_csv)
        # 读取农民CSV文件
        farmer_df = pd.read_csv(farmer_csv)

        # 确保两个数据集长度一致
        if len(landlord_df) != len(farmer_df):
            raise ValueError("地主和农民数据集行数不一致")

        # 计算数据点数量
        num_points = len(landlord_df)

        # 创建0-32小时的均匀分布的x轴
        x_hours = np.linspace(0, 32, num_points)

        # 绘制地主图表（包含胜率和ADP两个子图）
        plt.figure(figsize=(10, 8))

        # 绘制地主胜率曲线
        plt.subplot(2, 1, 1)
        plt.plot(x_hours, landlord_df['地主胜率'], 'b-o', linewidth=2, markersize=6)
        plt.title('地主胜率随时间变化曲线')
        plt.ylabel('地主胜率')
        plt.grid(True, linestyle='--', alpha=0.7)

        # 绘制地主ADP曲线
        plt.subplot(2, 1, 2)
        plt.plot(x_hours, landlord_df['地主ADP'], 'r-o', linewidth=2, markersize=6)
        plt.title('地主ADP随时间变化曲线')
        plt.xlabel('时间 (小时)')
        plt.ylabel('地主ADP')
        plt.grid(True, linestyle='--', alpha=0.7)

        # 自动调整布局
        plt.tight_layout()

        # 保存地主图像
        landlord_image_path = 'landlord_metrics_plot.png'
        plt.savefig(landlord_image_path, bbox_inches='tight')
        print(f"地主图像已成功保存到: {landlord_image_path}")

        # 如果使用交互式后端，则显示图像
        if matplotlib.get_backend() != 'Agg':
            plt.show()

        # 绘制农民图表（包含胜率和ADP两个子图）
        plt.figure(figsize=(10, 8))

        # 绘制农民胜率曲线
        plt.subplot(2, 1, 1)
        plt.plot(x_hours, farmer_df['农民胜率'], 'g-o', linewidth=2, markersize=6)
        plt.title('农民胜率随时间变化曲线')
        plt.ylabel('农民胜率')
        plt.grid(True, linestyle='--', alpha=0.7)

        # 绘制农民ADP曲线
        plt.subplot(2, 1, 2)
        plt.plot(x_hours, farmer_df['农民ADP'], 'm-o', linewidth=2, markersize=6)
        plt.title('农民ADP随时间变化曲线')
        plt.xlabel('时间 (小时)')
        plt.ylabel('农民ADP')
        plt.grid(True, linestyle='--', alpha=0.7)

        # 自动调整布局
        plt.tight_layout()

        # 保存农民图像
        farmer_image_path = 'farmer_metrics_plot.png'
        plt.savefig(farmer_image_path, bbox_inches='tight')
        print(f"农民图像已成功保存到: {farmer_image_path}")

        # 如果使用交互式后端，则显示图像
        if matplotlib.get_backend() != 'Agg':
            plt.show()

    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e.filename}")
    except ValueError as ve:
        print(f"错误: {ve}")
    except Exception as e:
        print(f"错误: 发生了一个未知错误: {e}")


if __name__ == "__main__":
    # CSV文件路径
    landlord_csv = 'landlord_evaluation_results.csv'
    farmer_csv = 'farmer_evaluation_results.csv'

    # 执行绘图
    plot_win_rate_and_adp(landlord_csv, farmer_csv)