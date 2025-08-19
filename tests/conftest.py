# -*- coding: utf-8 -*-
"""
pytest配置文件
设置测试环境和通用fixtures
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'modules'))

@pytest.fixture(scope="session")
def test_data_dir():
    """测试数据目录"""
    return os.path.join(project_root, 'tests', 'fixtures')

@pytest.fixture(scope="session")
def sample_stock_data():
    """生成样本股票数据"""
    # 生成一年的模拟股票数据
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # 过滤掉周末
    dates = dates[dates.dayofweek < 5]
    
    # 生成价格数据
    np.random.seed(42)  # 固定随机种子，确保测试可重复
    n_days = len(dates)
    
    # 生成收盘价（模拟真实股票价格走势）
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_days)  # 日收益率
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)
    
    # 生成OHLCV数据
    data = pd.DataFrame({
        'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n_days)
    }, index=dates)
    
    # 确保high >= low, high >= close, low <= close
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    return data

@pytest.fixture(scope="session")
def sample_features_data(sample_stock_data):
    """基于样本股票数据生成特征数据"""
    from modules.feature_engineer import FeatureEngineer
    
    # 创建特征工程师实例
    feature_engineer = FeatureEngineer()
    
    # 构建特征
    features = feature_engineer.build_features(sample_stock_data)
    
    return features

@pytest.fixture(scope="session")
def sample_model():
    """生成样本训练好的模型"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    # 创建简单的逻辑回归模型
    model = LogisticRegression(random_state=42)
    scaler = StandardScaler()
    
    return model, scaler

@pytest.fixture(scope="function")
def temp_test_dir(tmp_path):
    """临时测试目录"""
    return str(tmp_path)

@pytest.fixture(scope="session")
def mock_config():
    """模拟配置"""
    return {
        'data_dir': 'tests/fixtures',
        'features_dir': 'tests/fixtures/features',
        'models_dir': 'tests/fixtures/models',
        'results_dir': 'tests/fixtures/results',
        'commission': 0.0008,
        'initial_cash': 100000,
        'ml_threshold': 0.51
    }
