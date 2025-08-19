# 量化交易系统测试文档

## 📁 测试目录结构

```
tests/
├── __init__.py              # 测试包初始化
├── conftest.py              # pytest配置和fixtures
├── pytest.ini              # pytest配置文件
├── run_tests.py             # 测试运行脚本
├── README.md                # 本文档
├── unit/                    # 单元测试
│   ├── test_data_manager.py      # 数据管理器测试
│   ├── test_feature_engineer.py  # 特征工程师测试
│   ├── test_model_trainer.py     # 模型训练器测试
│   ├── test_backtest_analyzer.py # 回测分析器测试
│   └── test_prediction_system.py # 预测系统测试
├── integration/             # 集成测试
│   ├── test_trading_pipeline.py  # 交易流程测试
│   └── test_end_to_end.py        # 端到端测试
├── fixtures/                # 测试数据
│   ├── sample_data/         # 样本数据
│   └── mock_models/         # 模拟模型
└── reports/                 # 测试报告
    ├── test_reports/        # 测试结果报告
    └── coverage_reports/    # 代码覆盖率报告
```

## 🚀 快速开始

### 安装测试依赖

```bash
pip install pytest pytest-html pytest-cov pytest-mock
```

### 运行所有测试

```bash
# 使用测试脚本
python tests/run_tests.py

# 或直接使用pytest
pytest tests/ -v
```

### 运行特定类型测试

```bash
# 只运行单元测试
python tests/run_tests.py --unit

# 只运行集成测试
python tests/run_tests.py --integration

# 运行特定测试文件
python tests/run_tests.py --test tests/unit/test_data_manager.py
```

## 🧪 测试类型

### 单元测试 (Unit Tests)
- **位置**: `tests/unit/`
- **目的**: 测试单个函数或类的功能
- **特点**: 快速、独立、可重复
- **运行**: `pytest tests/unit/ -v`

### 集成测试 (Integration Tests)
- **位置**: `tests/integration/`
- **目的**: 测试模块间的协作
- **特点**: 测试真实的数据流和接口
- **运行**: `pytest tests/integration/ -v`

### 端到端测试 (End-to-End Tests)
- **位置**: `tests/integration/`
- **目的**: 测试完整的交易流程
- **特点**: 从数据获取到回测的完整验证
- **运行**: `pytest tests/integration/test_trading_pipeline.py -v`

## 📊 测试覆盖

### 核心模块测试
- ✅ **数据管理器** (`data_manager.py`)
- ✅ **特征工程师** (`feature_engineer.py`)
- ✅ **模型训练器** (`model_trainer.py`)
- ✅ **回测分析器** (`backtest_analyzer.py`)
- ✅ **预测系统** (`prediction_system.py`)

### 测试场景
- ✅ 数据加载和保存
- ✅ 特征构建和验证
- ✅ 模型训练和评估
- ✅ 回测执行和结果分析
- ✅ 预测生成和信号输出
- ✅ 错误处理和边界情况
- ✅ 性能指标和一致性检查

## 🔧 测试配置

### pytest配置
- **测试发现**: 自动发现`test_*.py`文件
- **输出格式**: 详细输出，彩色显示
- **警告处理**: 忽略常见警告
- **标记系统**: 支持单元/集成测试标记

### 测试数据
- **样本数据**: 生成模拟股票数据
- **模拟模型**: 创建测试用的机器学习模型
- **临时目录**: 每个测试使用独立的临时目录

## 📈 测试报告

### HTML报告
- **位置**: `tests/reports/test_report_YYYYMMDD_HHMMSS.html`
- **内容**: 详细的测试结果和失败信息
- **格式**: 自包含HTML，可直接在浏览器中查看

### 覆盖率报告
- **位置**: `tests/reports/coverage_report_YYYYMMDD_HHMMSS.html`
- **内容**: 代码覆盖率统计和未覆盖代码
- **目标**: 达到80%以上的代码覆盖率

## 🐛 故障排除

### 常见问题

1. **导入错误**
   ```bash
   # 确保在项目根目录运行
   cd /path/to/your/project
   python tests/run_tests.py
   ```

2. **依赖缺失**
   ```bash
   # 安装测试依赖
   pip install pytest pytest-html pytest-cov
   ```

3. **测试失败**
   ```bash
   # 查看详细错误信息
   pytest tests/ -v -s
   ```

### 调试技巧

1. **运行单个测试**
   ```bash
   pytest tests/unit/test_data_manager.py::TestDataManager::test_init -v -s
   ```

2. **查看测试输出**
   ```bash
   pytest tests/ -v -s --tb=long
   ```

3. **跳过慢速测试**
   ```bash
   pytest tests/ -m "not slow"
   ```

## 📝 添加新测试

### 创建单元测试
```python
# tests/unit/test_new_module.py
import pytest
from modules.new_module import NewClass

class TestNewClass:
    def test_new_function(self):
        """测试新功能"""
        obj = NewClass()
        result = obj.new_function()
        assert result is not None
```

### 创建集成测试
```python
# tests/integration/test_new_pipeline.py
import pytest

def test_new_pipeline():
    """测试新流程"""
    # 测试代码
    pass
```

### 运行新测试
```bash
# 运行特定测试文件
pytest tests/unit/test_new_module.py -v

# 运行所有新测试
pytest tests/ -k "new" -v
```

## 🎯 测试目标

### 质量目标
- **代码覆盖率**: ≥80%
- **测试通过率**: 100%
- **测试执行时间**: <5分钟
- **测试稳定性**: 可重复执行

### 功能验证
- ✅ 数据流程正确性
- ✅ 特征计算准确性
- ✅ 模型训练稳定性
- ✅ 回测结果合理性
- ✅ 预测系统可靠性

## 📞 支持

如有测试相关问题，请：
1. 查看测试输出和错误信息
2. 检查测试配置和依赖
3. 参考现有测试用例
4. 联系开发团队

---

**最后更新**: 2025-08-18
**测试状态**: ✅ 活跃维护
