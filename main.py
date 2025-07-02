import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy import stats
import os
from matplotlib.backends.backend_pdf import PdfPages


def load_xrd_data(file_path, file_format='csv'):
    """
    加载XRD数据，支持CSV、Excel、JSON、TXT格式。     fighting!
    :param file_path: 数据文件路径
    :param file_format: 数据文件格式 ('csv', 'excel', 'json', 'txt')
    :return: 返回pandas DataFrame的数据
    """
    if file_format == 'csv':
        data = pd.read_csv(file_path)
    elif file_format == 'excel':
        data = pd.read_excel(file_path)
    elif file_format == 'json':
        data = pd.read_json(file_path)
    elif file_format == 'txt':
        data = pd.read_csv(file_path, delimiter="\t")
    else:
        raise ValueError("Unsupported file format. Please use 'csv', 'excel', 'json', or 'txt'.")
    return data


def preprocess_data(data, noise_method='savgol', smooth_window=11, smooth_polyorder=3):
    """
    数据预处理，去噪和数据平滑。
    :param data: 原始数据
    :param noise_method: 噪声处理方法 ('savgol', 'moving_average', 'zscore')
    :param smooth_window: 平滑窗口大小
    :param smooth_polyorder: Savitzky-Golay平滑的多项式阶数
    :return: 预处理后的数据
    """
    if noise_method == 'savgol':
        smoothed_data = savgol_filter(data, window_length=smooth_window, polyorder=smooth_polyorder)
    elif noise_method == 'moving_average':
        smoothed_data = data.rolling(window=smooth_window).mean()
    elif noise_method == 'zscore':
        smoothed_data = stats.zscore(data)
    else:
        raise ValueError("Unsupported noise method. Please choose 'savgol', 'moving_average', or 'zscore'.")

    return smoothed_data


def detect_peaks(data, threshold=0.5, min_distance=10):
    """
    自动检测 XRD 图中的峰值。
    :param data: 处理后的强度数据
    :param threshold: 峰值的阈值，低于该阈值的点不会被认为是峰
    :param min_distance: 峰与峰之间的最小距离（数据点）
    :return: 峰值的位置和高度
    """
    peaks, properties = find_peaks(data, height=threshold, distance=min_distance)
    return peaks, properties['peak_heights']


def plot_xrd(data, output_file='xrd_plot.png', noise_method='savgol', smooth_window=11, smooth_polyorder=3,
             peak_detection=True):
    """
    绘制XRD图（2θ与强度的关系图）。
    :param data: 实验数据 (包含2theta和强度数据)
    :param output_file: 输出图像文件路径
    :param noise_method: 噪声处理方法
    :param smooth_window: 平滑窗口大小
    :param smooth_polyorder: Savitzky-Golay平滑的多项式阶数
    :param peak_detection: 是否执行峰值检测
    """
    smoothed_intensity = preprocess_data(data['intensity'], noise_method=noise_method, smooth_window=smooth_window,
                                         smooth_polyorder=smooth_polyorder)

    plt.figure(figsize=(8, 6))
    plt.plot(data['2theta'], smoothed_intensity, label='XRD', color='r')
    plt.xlabel('2θ (degrees)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('XRD Plot')

    if peak_detection:
        peaks, heights = detect_peaks(smoothed_intensity)
        plt.plot(data['2theta'][peaks], heights, "x", color='blue', label='Peaks')
        plt.legend()

    plt.grid(True)
    plt.savefig(output_file)
    plt.close()


def generate_xrd_report(file_path, file_format='csv', output_pdf='xrd_report.pdf', noise_method='savgol',
                        smooth_window=11, smooth_polyorder=3, peak_detection=True):
    """
    生成包含XRD图表和数据分析报告。
    :param file_path: XRD数据文件路径
    :param file_format: 数据文件格式 ('csv', 'excel', 'json', 'txt')
    :param output_pdf: 输出PDF报告路径
    :param noise_method: 噪声处理方法
    :param smooth_window: 平滑窗口大小
    :param smooth_polyorder: Savitzky-Golay平滑的多项式阶数
    :param peak_detection: 是否执行峰值检测
    """
    # 加载数据
    data = load_xrd_data(file_path=file_path, file_format=file_format)

    # 生成XRD图
    plot_xrd(data, output_file='xrd_plot.png', noise_method=noise_method, smooth_window=smooth_window,
             smooth_polyorder=smooth_polyorder, peak_detection=peak_detection)

    # 创建PDF报告
    with PdfPages(output_pdf) as pdf:
        plt.figure(figsize=(8, 6))
        plt.imshow(plt.imread('xrd_plot.png'))
        plt.axis('off')
        pdf.savefig()  # 保存当前图像
        plt.close()

        # 添加数据摘要
        plt.figure(figsize=(8, 6))
        plt.text(0.1, 0.9, f"XRD Data Summary", fontsize=14, ha='left')
        plt.text(0.1, 0.7, f"Number of Data Points: {len(data)}", fontsize=12, ha='left')
        plt.text(0.1, 0.5, f"Peak Detection: {'Enabled' if peak_detection else 'Disabled'}", fontsize=12, ha='left')
        pdf.savefig()  # 保存当前文本
        plt.close()

    print(f"XRD报告已生成并保存至：{output_pdf}")
