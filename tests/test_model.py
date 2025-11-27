import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from src.giems_lstm.model import LSTMNet, GRUNet, LSTMNetKAN, GRUNetKAN
from src.giems_lstm.model.kanlinear import KANLinear
from src.giems_lstm.data.dataset import WetlandDataset

# ==========================================
# 1. KANLinear Layer Tests (自定义层测试)
# ==========================================


def test_kan_linear_shape():
    batch_size = 16
    in_features = 32
    out_features = 8

    model = KANLinear(in_features, out_features)
    x = torch.randn(batch_size, in_features)

    out = model(x)
    assert out.shape == (batch_size, out_features), "KANLinear 输出形状不正确"


def test_kan_linear_update_grid():
    """测试 KAN 层的网格更新功能是否运行不报错"""
    model = KANLinear(in_features=10, out_features=5)
    x = torch.randn(32, 10)
    try:
        model.update_grid(x)
    except Exception as e:
        pytest.fail(f"KANLinear.update_grid 运行失败: {e}")


def test_kan_regularization():
    """测试正则化损失计算"""
    model = KANLinear(in_features=10, out_features=5)
    loss = model.regularization_loss()
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0, "正则化损失应非负"


# ==========================================
# 2. Model Forward Pass Tests (模型前向传播测试)
# ==========================================


@pytest.mark.parametrize("model_class", [LSTMNet, GRUNet, LSTMNetKAN, GRUNetKAN])
def test_models_forward_shape(model_class):
    batch_size = 8
    seq_length = 12
    input_dim = 10
    hidden_dim = 16
    output_dim = 1
    n_layers = 2
    device = torch.device("cpu")

    model = model_class(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        device=device,
    )

    # 构造输入 [Batch, Seq, Feature]
    x = torch.randn(batch_size, seq_length, input_dim).to(device)

    # 初始化 hidden
    h = model.init_hidden(batch_size)

    # 前向传播
    out, h_out = model(x, h)

    # 检查输出形状: 应该是 (Batch, output_dim) 因为代码中取了 out[:, -1]
    assert out.shape == (batch_size, output_dim), f"{model_class.__name__} 输出形状错误"

    # 检查输出范围 (因为用了 Sigmoid)
    assert (
        out.min() >= 0 and out.max() <= 1
    ), f"{model_class.__name__} 输出应在 [0, 1] 之间"


@pytest.mark.parametrize("model_class", [LSTMNet, GRUNet])
def test_models_backward(model_class):
    """测试梯度反向传播是否正常"""
    device = torch.device("cpu")
    model = model_class(5, 10, 1, 1, device)
    x = torch.randn(4, 10, 5)
    h = model.init_hidden(4)
    target = torch.rand(4, 1)

    out, _ = model(x, h)
    loss = nn.MSELoss()(out, target)
    loss.backward()

    # 检查是否有梯度产生
    for name, param in model.named_parameters():
        assert param.grad is not None, f"参数 {name} 没有梯度"


# ==========================================
# 3. Dataset Tests (数据处理测试)
# ==========================================


@pytest.fixture
def mock_data():
    """生成模拟的时序数据"""
    num_samples = 100
    features = np.random.rand(num_samples, 5)  # [T, D]
    target = np.random.rand(num_samples, 1)  # [T, 1]
    dates = pd.date_range(start="2020-01-01", periods=num_samples)
    return features, target, dates


def test_wetland_dataset_len(mock_data):
    features, target, dates = mock_data
    seq_length = 10

    dataset = WetlandDataset(features, dates.tolist(), target, seq_length=seq_length)

    # 预期窗口数量 = 总长度 - 序列长度 + 1
    expected_len = len(features) - seq_length + 1
    assert len(dataset) == expected_len, "Dataset 长度计算错误"


def test_wetland_dataset_item(mock_data):
    features, target, dates = mock_data
    seq_length = 5
    dataset = WetlandDataset(features, dates.tolist(), target, seq_length=seq_length)

    # 获取第一个样本
    item = dataset[0]
    # 检查返回格式 (features_window, target_window)
    assert len(item) == 2

    x_window, y_window = item
    assert x_window.shape == (seq_length, 5)
    # y 应该是单个时间步的值（根据代码逻辑 target_window = self.target[target_index]）
    assert y_window.shape == (1,)


def test_dataset_predict_mode(mock_data):
    features, target, dates = mock_data
    dataset = WetlandDataset(features, dates.tolist(), seq_length=10, predict_mode=True)

    item = dataset[0]
    # 预测模式下只返回 feature_window
    assert isinstance(item, np.ndarray)
    assert item.shape == (10, 5)


def test_nan_handling():
    """测试包含 NaN 的目标值是否被跳过"""
    features = np.random.rand(20, 2)
    target = np.random.rand(20, 1)
    # 在第 11 个点设置 NaN (影响索引为 11-seq_length+1 的窗口)
    target[10] = np.nan
    dates = pd.date_range("20200101", periods=20).tolist()

    seq_length = 5
    dataset = WetlandDataset(features, dates, target, seq_length=seq_length)

    # 理论总数
    total_windows = 20 - 5 + 1
    # 实际数量应该少于理论总数，因为有一个 NaN target
    assert len(dataset) < total_windows
