# AI量化交易学习平台

一个专为初学者设计的AI量化交易学习平台，采用分层架构设计，支持完整的量化交易流程：从数据获取到模型训练，从回测分析到未来预测。

## 🚀 项目特色

- **分层架构设计**：清晰的数据层、特征层、模型层、预测层分离
- **完整交易流程**：数据获取 → 特征工程 → 模型训练 → 回测分析 → 历史验证 → 未来预测
- **Web可视化界面**：直观的操作界面，实时进度显示
- **多股票支持**：支持批量处理多只股票
- **机器学习集成**：集成Logistic Regression等经典算法
- **技术指标丰富**：包含移动平均、RSI、MACD等常用技术指标

## 🏗️ 系统架构

```
AI量化交易学习平台
├── 数据层 (Data Layer)
│   ├── 数据获取与清洗 (DataManager)
│   └── 本地数据存储
├── 特征层 (Feature Layer)
│   ├── 技术指标构建 (FeatureEngineer)
│   └── 特征数据管理
├── 模型层 (Model Layer)
│   ├── 模型训练与保存 (ModelTrainer)
│   └── 交叉验证与评估
├── 预测层 (Prediction Layer)
│   ├── 历史预测验证 (HistoricalPredictor)
│   └── 未来预测 (FuturePredictor)
├── 回测层 (Backtest Layer)
│   ├── 策略回测 (BacktestAnalyzer)
│   └── 性能指标分析
└── Web界面层
    ├── Flask后端API
    └── 响应式前端界面
```

## 📋 功能模块

### 1. 数据获取与清洗
- 支持多数据源：akshare、baostock
- 自动数据清洗和标准化
- 本地数据存储管理
- 数据状态监控

### 2. 特征工程
- 技术指标构建：
  - 移动平均线 (MA)
  - 相对强弱指数 (RSI)
  - MACD指标
  - 布林带 (Bollinger Bands)
  - 随机指标 (Stochastic)
  - 威廉指标 (Williams %R)
- 特征数据持久化
- 特征质量检查

### 3. 模型训练与保存
- 机器学习模型训练
- 时间序列数据分割
- 交叉验证评估
- 模型性能指标
- 模型持久化存储

### 4. 历史预测验证
- 使用未参与训练的数据验证模型
- 预测准确率评估
- 模型泛化能力测试
- 验证结果分析

### 5. 未来预测
- 基于最新特征预测未来1-5天
- 生成交易信号 (BUY/SELL/HOLD)
- 置信度评估
- 预测结果保存

### 6. 回测分析
- 历史策略回测
- 性能指标计算：
  - 总收益率
  - 年化收益率
  - 最大回撤
  - 夏普比率
  - 胜率统计
- 交易记录分析

## 🛠️ 技术栈

### 后端
- **Python 3.8+**：核心编程语言
- **Flask**：Web框架
- **pandas**：数据处理
- **numpy**：数值计算
- **scikit-learn**：机器学习
- **backtrader**：回测框架

### 前端
- **HTML5/CSS3**：页面结构
- **JavaScript (ES6+)**：交互逻辑
- **Bootstrap 5**：UI框架
- **Chart.js**：图表可视化

### 数据源
- **akshare**：开源金融数据接口
- **baostock**：证券宝数据接口

## 📦 安装部署

### 环境要求
- Python 3.8+
- pip 包管理器
- 现代浏览器 (Chrome, Firefox, Safari, Edge)

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/halouxiaoyu/QuantitativeLearning.git
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **启动服务**
```bash
python web_app.py
```

4. **访问平台**
打开浏览器访问：`http://localhost:5002`

## 📖 使用指南

### 快速开始

1. **数据获取**
   - 选择股票池（如银行股）
   - 设置时间范围（建议2021-2025）
   - 点击"开始下载"

2. **特征构建**
   - 选择特征类型
   - 点击"开始构建特征"

3. **模型训练**
   - 设置训练参数
   - 点击"开始训练模型"

4. **回测分析**
   - 设置回测参数
   - 点击"开始回测"

5. **历史验证**
   - 设置验证时间范围
   - 点击"开始历史验证"

6. **未来预测**
   - 输入股票代码
   - 设置预测天数（1-5天）
   - 点击"开始未来预测"

### 工作流程

```
数据获取 → 特征工程 → 模型训练 → 回测分析 → 历史验证 → 未来预测
    ↓           ↓         ↓         ↓         ↓         ↓
  原始数据 → 技术指标 → 训练模型 → 策略回测 → 模型验证 → 交易信号
```

## 🔧 配置说明

### 默认时间配置
- **训练时间**：2021-01-01 到 2024-12-31
- **验证时间**：2025-01-01 到 2025-12-31
- **测试比例**：训练数据的20%用于内部测试

### 模型参数
- **算法**：Logistic Regression
- **交叉验证**：5折交叉验证
- **随机种子**：42（可配置）

### 回测参数
- **初始资金**：100,000元
- **手续费率**：0.03%
- **ML信号阈值**：0.51

## 📊 测试覆盖

项目包含完整的测试框架：
- **单元测试**：各模块独立功能测试 (28个测试)
- **集成测试**：模块间协作测试 (3个测试)
- **功能测试**：核心功能验证测试 (3个测试)
- **总测试数量**：34个测试，全部通过
- **代码覆盖率**：28% (核心功能模块覆盖率较高)

运行测试：
```bash
# 运行所有测试
python tests/run_tests.py

# 运行特定测试
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v

# 生成覆盖率报告
python -m pytest --cov=modules --cov-report=html tests/
```

模块覆盖率详情：
- **FeatureEngineer**: 73% (核心功能完整测试)
- **DataManager**: 37% (主要功能已测试)
- **BacktestAnalyzer**: 23% (基础功能已测试)
- **ModelTrainer**: 13% (核心训练逻辑已测试)
- **HistoricalPredictor**: 11% (接口功能已测试)
- **FuturePredictor**: 13% (接口功能已测试)

## 🚨 注意事项

1. **数据质量**：确保数据源稳定，数据质量良好
2. **模型风险**：机器学习模型存在过拟合风险，建议定期重新训练
3. **回测偏差**：历史回测结果不代表未来表现
4. **资金管理**：实际交易请做好风险控制和资金管理

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 贡献流程
1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 联系方式

- 项目主页：[GitHub Repository](https://github.com/halouxiaoyu/QuantitativeLearning)
- 问题反馈：[Issues](https://github.com/halouxiaoyu/QuantitativeLearning/issues)

## 🙏 致谢

感谢以下开源项目的支持：
- [akshare](https://github.com/akfamily/akshare) - 开源金融数据接口
- [baostock](http://baostock.com/baostock/index.html) - 证券宝数据接口
- [scikit-learn](https://scikit-learn.org/) - 机器学习库
- [backtrader](https://www.backtrader.com/) - 回测框架
- [Flask](https://flask.palletsprojects.com/) - Web框架

---

⭐ 如果这个项目对你有帮助，请给它一个星标！
