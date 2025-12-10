# @Author：青松
import random
import sys

import numpy as np
import pandas as pd
import torch
import lightning as L
from colorama import Fore, Style
from prettytable import PrettyTable


def show_version():
    print("Python version: ", sys.version)
    print("Pytorch version: ", torch.__version__)
    print("PyTorch Lightning version: ", L.__version__)


def show_table(data):
    # 设置浮点数显示格式，避免使用科学记数法影响阅读体验
    pd.set_option('display.float_format', '{:.4f}'.format)
    df = pd.DataFrame(data)
    display(df)


def print_table(table_name, field_names: list, data: list):
    # 编码解码示例表
    table = PrettyTable()
    table.field_names = field_names
    for data_item in data:
        table.add_row(data_item)
    print(f"\n{table_name}：")
    print(table)


def print_red(text):
    print(Fore.RED + text + Style.RESET_ALL)


def print_regression_results(inputs, true_values, predictions, table_name="预测结果",
                             field_names=["输入", "真实值", "预测值", "预测偏差", "预测精度"]):
    """
    打印回归预测的结果
    """

    # 处理 tensor 类型的预测结果
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.tolist()

    # 准备表格数据
    table_data = []

    for i in range(len(inputs)):
        table_data.append([
            inputs[i],
            round(true_values[i], 4),
            round(predictions[i], 4),
            f"{predictions[i] - true_values[i]:.4f}",
            f"{predictions[i] * 100 / true_values[i]:.2f}%"
        ])

    print_table(
        table_name,
        field_names,
        table_data
    )


def print_classification_results(inputs, true_labels, predictions, probabilities, label_map):
    """
    打印文本分类预测结果，包含预测标签和真实标签的对比

    Args:
        inputs: 待预测的样本列表
        true_labels: 真实标签列表（数字形式）
        predictions: 模型预测结果（数字形式）
        probabilities: 预测概率列表
        label_map: 标签映射字典
    """

    # 处理 tensor 类型的预测结果
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.tolist()
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.tolist()

    # 准备表格数据
    table_data = []
    correct_count = 0

    for i, text in enumerate(inputs):
        # 获取真实标签和预测标签名称
        true_label_name = label_map[true_labels[i]]
        pred_label_name = label_map[predictions[i]]

        # 判断预测是否正确
        is_correct = (true_labels[i] == predictions[i])
        if is_correct:
            correct_count += 1
            mark = '\033[92m' + '☑' + '\033[0m'  # 绿色勾号表示正确
        else:
            mark = '\033[91m' + '☒' + '\033[0m'  # 红色叉号表示错误

        # 处理长文本显示
        display_text = text[:30] + "..." if len(text) > 30 else text

        # 获取最高概率
        if isinstance(probabilities[i], (list, tuple)):
            max_prob = max(probabilities[i])
        else:
            max_prob = probabilities[i].max().item()

        # 添加行数据
        table_data.append([
            display_text,
            true_label_name,
            pred_label_name,
            f"{max_prob:.4f}",
            mark
        ])

    # 使用 print_table 方法打印结果
    accuracy = correct_count / len(inputs) if len(inputs) > 0 else 0
    print_table(
        f"分类预测结果 (准确率: {correct_count}/{len(inputs)} = {accuracy * 100:.2f}%)",
        ["输入", "真实标签", "预测标签", "最高概率", "标记"],
        table_data
    )


def set_seed(seed=1024):
    # 设置 Python 的随机种子
    random.seed(seed)

    # 设置 NumPy 的随机种子
    np.random.seed(seed)

    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # 确保结果的可重复性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
