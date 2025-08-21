#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
模型训练与保存模块
支持多股票模型训练、验证和保存
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 机器学习相关

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import json

# 尝试导入XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost未安装，将使用随机森林作为备选")

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, features_dir, models_dir, results_dir):
        self.features_dir = features_dir
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        # 默认时间划分配置
        self.default_time_config = {
            'training_start': '2021-01-01',    # 训练开始时间
            'training_end': '2024-12-31',      # 训练结束时间
            'validation_start': '2025-01-01',  # 验证开始时间
            'validation_end': '2025-12-31',    # 验证结束时间
            'test_ratio': 0.2                  # 测试集比例
        }
        
        # 创建必要的目录
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'training_reports'), exist_ok=True)
    
    def load_features(self, stock_code):
        """加载特征数据"""
        stock_dir = os.path.join(self.features_dir, stock_code)
        if not os.path.exists(stock_dir):
            raise FileNotFoundError(f"未找到特征目录: {stock_code}")
        
        # 查找最新的特征文件
        feature_files = [f for f in os.listdir(stock_dir) if f.endswith('.csv') and 'features' in f]
        if not feature_files:
            raise FileNotFoundError(f"未找到特征文件: {stock_code}")
        
        latest_file = sorted(feature_files)[-1]
        file_path = os.path.join(stock_dir, latest_file)
        
        print(f"📂 加载特征: {os.path.basename(latest_file)}")
        
        try:
            feat = pd.read_csv(file_path, index_col=0, parse_dates=True)
            print(f"   ✅ 特征加载成功: {len(feat)} 条记录")
            print(f"   📅 数据时间范围: {feat.index.min().strftime('%Y-%m-%d')} 到 {feat.index.max().strftime('%Y-%m-%d')}")
            return feat
        except Exception as e:
            raise Exception(f"特征加载失败: {e}")
    
    def prepare_training_data(self, stock_code_or_data, start_date=None, end_date=None, test_ratio=None, return_split=True):
        """准备训练数据，使用默认时间划分"""
        print(f"🔧 准备训练数据: {stock_code_or_data}")
        print("=" * 50)
        
        # 检查输入类型
        if isinstance(stock_code_or_data, str):
            # 如果是股票代码，加载特征数据
            feat = self.load_features(stock_code_or_data)
        elif hasattr(stock_code_or_data, 'columns'):
            # 如果是DataFrame，直接使用
            feat = stock_code_or_data
        else:
            raise ValueError("输入必须是股票代码字符串或DataFrame")
        
        # 使用默认时间配置
        if start_date is None:
            start_date = self.default_time_config['training_start']
        if end_date is None:
            end_date = self.default_time_config['training_end']
        if test_ratio is None:
            test_ratio = self.default_time_config['test_ratio']
        
        print(f"📅 训练时间范围: {start_date} 到 {end_date}")
        print(f"📊 测试集比例: {test_ratio:.1%}")
        
        # 过滤训练数据
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # 确保数据在指定范围内
        available_start = feat.index.min()
        available_end = feat.index.max()
        
        if start_date < available_start:
            print(f"⚠️  请求的开始时间 {start_date.strftime('%Y-%m-%d')} 早于可用数据 {available_start.strftime('%Y-%m-%d')}，使用可用数据开始时间")
            start_date = available_start
        
        if end_date > available_end:
            print(f"⚠️  请求的结束时间 {end_date.strftime('%Y-%m-%d')} 晚于可用数据 {available_end.strftime('%Y-%m-%d')}，使用可用数据结束时间")
            end_date = available_end
        
        # 过滤训练数据
        training_data = feat[(feat.index >= start_date) & (feat.index <= end_date)].copy()
        
        if len(training_data) == 0:
            raise ValueError(f"在指定时间范围内没有找到数据: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
        
        print(f"📈 训练数据: {len(training_data)} 条记录")
        print(f"📅 实际训练范围: {training_data.index.min().strftime('%Y-%m-%d')} 到 {training_data.index.max().strftime('%Y-%m-%d')}")
        
        # 准备特征和标签
        feature_cols = [col for col in training_data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        if len(feature_cols) == 0:
            raise ValueError("没有找到有效的特征列")
        
        X = training_data[feature_cols].values
        
        # 使用特征文件中的标签列（第二天涨跌幅，0.1%阈值）
        y = training_data['label'].values
        
        # 移除包含NaN的行
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"🔍 有效训练样本: {len(X)} 条")
        print(f"📊 特征维度: {X.shape}")
        print(f"🎯 标签分布: 上涨 {np.sum(y)} ({np.sum(y)/len(y):.1%}), 下跌 {np.sum(~y)} ({np.sum(~y)/len(y):.1%})")
        
        # 时间序列分割（保持时间顺序）
        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"📚 训练集: {len(X_train)} 条 ({len(X_train)/len(X):.1%})")
        print(f"🧪 测试集: {len(X_test)} 条 ({len(X_test)/len(X):.1%})")
        
        if return_split:
            return X_train, X_test, y_train, y_test, feature_cols
        else:
            # 返回合并后的数据，用于测试
            X_combined = np.vstack([X_train, X_test])
            y_combined = np.concatenate([y_train, y_test])
            dates_combined = np.concatenate([feat.index[:len(X_train)], feat.index[len(X_train):len(X_train)+len(X_test)]])
            return X_combined, y_combined, feature_cols, dates_combined
    
    def get_available_algorithms(self):
        """获取可用的算法列表"""
        algorithms = {
            'random_forest': {
                'name': '随机森林',
                'description': '集成学习，抗过拟合，适合复杂数据',
                'available': True
            },
            'xgboost': {
                'name': 'XGBoost',
                'description': '梯度提升，性能优秀，适合结构化数据',
                'available': XGBOOST_AVAILABLE
            }
        }
        return algorithms
    
    def create_model(self, algorithm='random_forest', random_seed=42):
        """根据算法名称创建模型
        
        Args:
            algorithm: 算法名称 ('random_forest', 'xgboost')
            random_seed: 随机种子
            
        Returns:
            训练好的模型
        """
        if algorithm == 'random_forest':
            print("🌲 使用随机森林模型")
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_seed,
                n_jobs=-1
            )
        
        elif algorithm == 'xgboost' and XGBOOST_AVAILABLE:
            print("🚀 使用XGBoost模型")
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_seed,
                n_jobs=-1
            )
        
        else:
            if algorithm == 'xgboost' and not XGBOOST_AVAILABLE:
                print("⚠️  XGBoost不可用，自动切换到随机森林")
                algorithm = 'random_forest'
            else:
                print(f"⚠️  未知算法 '{algorithm}'，使用随机森林")
                algorithm = 'random_forest'
            
            return self.create_model('random_forest', random_seed)
    
    def train_model(self, stock_code, algorithm='random_forest', cv_folds=5, start_date=None, end_date=None, test_ratio=None, random_seed=42):
        """训练模型，使用指定的算法和时间划分"""
        print(f"🤖 开始训练模型: {stock_code}")
        print(f"🧠 使用算法: {algorithm}")
        print("=" * 60)
        
        try:
            # 1. 准备训练数据
            X_train, X_test, y_train, y_test, feature_cols = self.prepare_training_data(
                stock_code, start_date, end_date, test_ratio
            )
            
            # 2. 数据标准化
            print("📏 数据标准化...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 3. 模型训练
            print(f"🎯 训练{self.get_available_algorithms()[algorithm]['name']}模型...")
            
            # 创建指定算法的模型
            model = self.create_model(algorithm, random_seed)
            
            # 时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring='accuracy')
            
            print(f"📊 交叉验证准确率: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # 在完整训练集上训练最终模型
            model.fit(X_train_scaled, y_train)
            
            # 4. 模型评估
            print("📈 模型评估...")
            
            # 训练集评估
            y_train_pred = model.predict(X_train_scaled)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred)
            
            # 测试集评估
            y_test_pred = model.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            
            print(f"📚 训练集 - 准确率: {train_accuracy:.3f}, F1分数: {train_f1:.3f}")
            print(f"🧪 测试集 - 准确率: {test_accuracy:.3f}, F1分数: {test_f1:.3f}")
            
            # 5. 保存模型和结果
            self._save_model_and_results(stock_code, model, scaler, feature_cols, {
                'training_date': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'algorithm': algorithm,
                'cv_scores': {
                    'mean': float(cv_scores.mean()),
                    'std': float(cv_scores.std()),
                    'scores': cv_scores.tolist()
                },
                'training_metrics': {
                    'train_accuracy': float(train_accuracy),
                    'train_f1': float(train_f1),
                    'test_accuracy': float(test_accuracy),
                    'test_f1': float(test_f1)
                },
                'data_info': {
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'feature_count': len(feature_cols),
                    'training_start': start_date if start_date else self.default_time_config['training_start'],
                    'training_end': end_date if end_date else self.default_time_config['training_end']
                }
            })
            
            # 6. 生成训练报告
            self._generate_training_report(stock_code, y_test, y_test_pred, feature_cols)
            
            print(f"✅ 模型训练完成: {stock_code}")
            return True
            
        except Exception as e:
            print(f"❌ 模型训练失败: {e}")
            return False
    
    def get_validation_data_info(self, stock_code):
        """获取验证数据信息（2025年数据）"""
        print(f"🔍 获取验证数据信息: {stock_code}")
        
        feat = self.load_features(stock_code)
        
        validation_start = pd.to_datetime(self.default_time_config['validation_start'])
        validation_end = pd.to_datetime(self.default_time_config['validation_end'])
        
        # 检查是否有验证数据
        available_start = feat.index.min()
        available_end = feat.index.max()
        
        if validation_end < available_start or validation_start > available_end:
            print(f"⚠️  验证时间范围 {validation_start.strftime('%Y-%m-%d')} 到 {validation_end.strftime('%Y-%m-%d')} 与可用数据不重叠")
            return None
        
        # 过滤验证数据
        validation_data = feat[(feat.index >= validation_start) & (feat.index <= validation_end)].copy()
        
        if len(validation_data) == 0:
            print(f"⚠️  在验证时间范围内没有找到数据")
            return None
        
        info = {
            'validation_start': validation_data.index.min().strftime('%Y-%m-%d'),
            'validation_end': validation_data.index.max().strftime('%Y-%m-%d'),
            'sample_count': len(validation_data),
            'available': True
        }
        
        print(f"📊 验证数据: {info['sample_count']} 条记录")
        print(f"📅 验证时间范围: {info['validation_start']} 到 {info['validation_end']}")
        
        return info
    
    def _save_model_and_results(self, stock_code, model, scaler, feature_cols, model_info):
        """保存模型和相关信息"""
        stock_model_dir = os.path.join(self.models_dir, stock_code)
        os.makedirs(stock_model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存模型
        model_path = os.path.join(stock_model_dir, f"{stock_code}_model_{timestamp}.pkl")
        joblib.dump(model, model_path)
        print(f"💾 模型已保存: {os.path.basename(model_path)}")
        
        # 保存标准化器
        scaler_path = os.path.join(stock_model_dir, f"{stock_code}_scaler_{timestamp}.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"💾 标准化器已保存: {os.path.basename(scaler_path)}")
        
        # 保存模型信息
        model_info['feature_names'] = feature_cols
        info_path = os.path.join(stock_model_dir, f"{stock_code}_info_{timestamp}.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2, default=str)
        print(f"💾 模型信息已保存: {os.path.basename(info_path)}")
    
    def _generate_training_report(self, stock_code, y_true, y_pred, feature_cols):
        """生成训练报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.results_dir, 'training_reports', f"{stock_code}_training_report_{timestamp}.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"模型训练报告 - {stock_code}\n")
            f.write("=" * 50 + "\n")
            f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"特征数量: {len(feature_cols)}\n")
            f.write(f"测试集样本数: {len(y_true)}\n")
            f.write(f"测试集准确率: {accuracy_score(y_true, y_pred):.3f}\n")
            f.write(f"测试集F1分数: {f1_score(y_true, y_pred):.3f}\n")
            f.write("\n分类报告:\n")
            f.write(classification_report(y_true, y_pred))
            f.write("\n混淆矩阵:\n")
            f.write(str(confusion_matrix(y_true, y_pred)))
        
        print(f"📄 训练报告已生成: {os.path.basename(report_path)}")
    
    def batch_train_models(self, stock_list=None, algorithm='random_forest', cv_folds=5, start_date=None, end_date=None, random_seed=42):
        """批量训练多个股票的模型
        
        Args:
            stock_list: 股票列表
            algorithm: 算法选择 ('logistic', 'random_forest', 'xgboost')
            cv_folds: 交叉验证折数
            start_date: 训练开始日期
            end_date: 训练结束日期
            random_seed: 随机种子
        """
        if stock_list is None:
            # 从特征目录获取股票列表
            if os.path.exists(self.features_dir):
                stock_list = [d for d in os.listdir(self.features_dir) if os.path.isdir(os.path.join(self.features_dir, d))]
            else:
                print("❌ 没有找到特征目录")
                return {}
        
        if not stock_list:
            print("❌ 没有股票可训练")
            return {}
        
        # 使用传入的参数或默认值
        training_start = start_date if start_date else self.default_time_config['training_start']
        training_end = end_date if end_date else self.default_time_config['training_end']
        
        print(f"🚀 批量训练模型: {len(stock_list)} 只股票")
        print("=" * 60)
        print(f"🧠 使用算法: {algorithm}")
        print(f"📅 训练时间: {training_start} 到 {training_end}")
        print(f"📅 验证时间: {self.default_time_config['validation_start']} 到 {self.default_time_config['validation_end']}")
        print(f"🎲 随机种子: {random_seed}")
        print("=" * 60)
        
        results = {}
        
        for i, stock_code in enumerate(stock_list, 1):
            print(f"\n🤖 [{i}/{len(stock_list)}] 训练股票: {stock_code}")
            print("-" * 40)
            
            try:
                # 检查验证数据可用性
                validation_info = self.get_validation_data_info(stock_code)
                if not validation_info or not validation_info['available']:
                    print(f"⚠️  {stock_code}: 验证数据不可用，跳过训练")
                    results[stock_code] = {
                        'success': False,
                        'error': '验证数据不可用'
                    }
                    continue
                
                success = self.train_model(stock_code, algorithm, cv_folds, training_start, training_end, random_seed=random_seed)
                
                if success:
                    results[stock_code] = {
                        'success': True,
                        'validation_info': validation_info
                    }
                else:
                    results[stock_code] = {
                        'success': False,
                        'error': '训练失败'
                    }
                    
            except Exception as e:
                print(f"❌ {stock_code}: 训练异常 - {e}")
                results[stock_code] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results

def main():
    """主函数"""
    print("🤖 模型训练与保存模块")
    print("=" * 60)
    
    # 创建模型训练器
    # 假设特征目录、模型目录和结果目录已经存在
    features_dir = "features"
    models_dir = "models"
    results_dir = "results"
    
    mt = ModelTrainer(features_dir, models_dir, results_dir)
    
    print("🔧 系统状态:")
    print(f"   特征目录: {os.path.abspath(mt.features_dir)}")
    print(f"   模型目录: {os.path.abspath(mt.models_dir)}")
    print(f"   结果目录: {os.path.abspath(mt.results_dir)}")
    
    # 检查是否有特征数据
    if not os.path.exists(mt.features_dir):
        print("\n❌ 没有找到特征目录!")
        print("   请先运行特征工程模块构建特征")
        return
    
    # 获取可用的股票
    stock_dirs = [d for d in os.listdir(mt.features_dir) if os.path.isdir(os.path.join(mt.features_dir, d))]
    
    if not stock_dirs:
        print("\n❌ 没有找到特征数据!")
        print("   请先运行特征工程模块构建特征")
        return
    
    print(f"\n📊 发现 {len(stock_dirs)} 只股票的特征:")
    for stock_dir in stock_dirs:
        print(f"   • {stock_dir}")
    
    # 开始训练模型
    print(f"\n🚀 开始训练模型...")
    results = mt.batch_train_models(stock_list=stock_dirs)
    
    # 显示模型摘要
    print(f"\n�� 模型训练完成，查看摘要:")
    # mt.get_model_summary() # This method is removed from the new_code, so we'll skip this for now.

if __name__ == "__main__":
    main()
