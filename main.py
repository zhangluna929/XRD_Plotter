import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy import stats
import os
from matplotlib.backends.backend_pdf import PdfPages

def load_xrd_data(file_path, file_format='csv'):
    """
    加载XRD数据，支持CSV、Excel、JSON、TXT格式。
    :param file_path: 数据文件路径
    :param file_format: 数据文件格式 ('csv', 'excel', 'json', 'txt')
    :return: pandas DataFrame 格式的数据
    """
    file_handlers = {
        'csv': pd.read_csv,
        'excel': pd.read_excel,
        'json': pd.read_json,
        'txt': lambda x: pd.read_csv(x, delimiter="\t")
    }

    if file_format not in file_handlers:
        raise ValueError(f"不支持该格式：'{file_format}'，支持的格式有：'csv', 'excel', 'json', 'txt'。")

    try:
        return file_handlers[file_format](file_path)
    except Exception as e:
        print(f"加载数据时出错: {e}")
        raise


def preprocess_data(data, method='savgol', window_size=11, poly_order=3):
    """
    数据预处理。
    :param data: 原始数据
    :param method: 噪声处理方法 ('savgol', 'moving_average', 'zscore')
    :param window_size: 窗口大小
    :param poly_order: Savitzky-Golay滤波器的多项式阶数
    :return: 处理后的数据
    """
    if method == 'savgol':
        return savgol_filter(data, window_length=window_size, polyorder=poly_order)
    elif method == 'moving_average':
        return data.rolling(window=window_size).mean()
    elif method == 'zscore':
        return stats.zscore(data)
    else:
        raise ValueError(f"方法'{method}'不被支持，请选择 'savgol', 'moving_average', 或 'zscore'.")


def find_xrd_peaks(data, threshold=0.5, min_distance=10):
    """
    检测XRD图谱中的峰值位置和强度。
    :param data: 吸光度数据
    :param threshold: 峰值高度阈值
    :param min_distance: 峰值间的最小距离
    :return: 峰值位置和高度
    """
    peaks, properties = find_peaks(data, height=threshold, distance=min_distance)
    return peaks, properties['peak_heights']


def plot_xrd(data, output_file='xrd_plot.png', method='savgol', window_size=11, poly_order=3, detect_peaks=True):
    """
    绘制XRD图谱（2θ与强度的关系图）。
    :param data: 数据集（包含2θ和强度）
    :param output_file: 输出图像文件路径
    :param method: 数据平滑方法
    :param window_size: 平滑窗口大小
    :param poly_order: Savitzky-Golay平滑的多项式阶数
    :param detect_peaks: 是否执行峰值检测
    """
    smoothed_data = preprocess_data(data['intensity'], method=method, window_size=window_size, poly_order=poly_order)

    plt.figure(figsize=(10, 6))
    plt.plot(data['2theta'], smoothed_data, color='red', label='XRD Spectrum')
    plt.xlabel('2θ (degrees)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('XRD Spectrum')

    if detect_peaks:
        peaks, heights = find_xrd_peaks(smoothed_data)
        plt.plot(data['2theta'][peaks], heights, "x", color='blue', label='Detected Peaks')
        plt.legend()

    plt.grid(True)
    plt.savefig(output_file)
    plt.close()


def generate_xrd_report(file_path, file_format='csv', output_pdf='xrd_report.pdf', method='savgol',
                         window_size=11, poly_order=3, detect_peaks=True):
    """
    生成XRD图谱及数据报告（good luck）。
    :param file_path: XRD数据文件路径
    :param file_format: 数据格式 ('csv', 'excel', 'json', 'txt')
    :param output_pdf: 输出的PDF报告路径
    :param method: 噪声处理方法
    :param window_size: 平滑窗口大小
    :param poly_order: Savitzky-Golay平滑的多项式阶数
    :param detect_peaks: 是否执行峰值检测
    """
    data = load_xrd_data(file_path, file_format)

    plot_xrd(data, output_file='xrd_plot.png', method=method, window_size=window_size, poly_order=poly_order,
             detect_peaks=detect_peaks)

    with PdfPages(output_pdf) as pdf:
        # 插入XRD图
        plt.figure(figsize=(10, 6))
        plt.imshow(plt.imread('xrd_plot.png'))
        plt.axis('off')
        pdf.savefig()  # 保存图像
        plt.close()

        # 数据摘要
        plt.figure(figsize=(10, 6))
        plt.text(0.1, 0.9, f"XRD Data Summary", fontsize=16, ha='left')
        plt.text(0.1, 0.7, f"Total Data Points: {len(data)}", fontsize=14, ha='left')
        plt.text(0.1, 0.5, f"Peak Detection: {'Enabled' if detect_peaks else 'Disabled'}", fontsize=14, ha='left')
        plt.text(0.1, 0.3, f"Smoothing Method: {method}", fontsize=14, ha='left')
        pdf.savefig()  # 保存文本摘要
        plt.close()

    print(f"XRD报告已生成并保存在：{output_pdf}")
