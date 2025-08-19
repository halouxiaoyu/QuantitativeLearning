# -*- coding: utf-8 -*-
"""
历史预测验证器
用于在历史数据上验证模型效果，不是真正的未来预测
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from modules.model_trainer import ModelTrainer


class HistoricalPredictor:
    """历史预测验证器 - 在历史数据上验证模型效果"""
    
    def __init__(self, features_dir, models_dir, results_dir):
        self.features_dir = features_dir
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.prediction_history = {}
        
        # 默认时间划分配置（与ModelTrainer保持一致）
        self.default_time_config = {
            'training_start': '2021-01-01',    # 训练开始时间
            'training_end': '2024-12-31',      # 训练结束时间
            'validation_start': '2025-01-01',  # 验证开始时间（默认）
            'validation_end': '2025-12-31',    # 验证结束时间（默认）
        }
        
        # 创建必要的目录
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'reports'), exist_ok=True)
    
    def load_model(self, stock_code):
        """加载训练好的模型"""
        model_dir = os.path.join(self.models_dir, stock_code)
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"未找到模型目录: {stock_code}")
        
        # 查找最新的模型文件
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and 'model' in f]
        if not model_files:
            raise FileNotFoundError(f"未找到模型文件: {stock_code}")
        
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(model_dir, latest_model)
        
        # 加载模型
        import joblib
        model = joblib.load(model_path)
        
        # 查找对应的标准化器
        scaler_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and 'scaler' in f]
        scaler = None
        if scaler_files:
            latest_scaler = sorted(scaler_files)[-1]
            scaler_path = os.path.join(model_dir, latest_scaler)
            scaler = joblib.load(scaler_path)
        
        # 查找模型信息文件
        info_files = [f for f in os.listdir(model_dir) if f.endswith('.json') and 'info' in f]
        model_info = {}
        if info_files:
            latest_info = sorted(info_files)[-1]
            info_path = os.path.join(model_dir, latest_info)
            import json
            with open(info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
        
        return model, scaler, model_info
    
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
    
    def validate_historical_predictions(self, stock_code, start_date=None, end_date=None):
        """在历史数据上验证模型效果，默认使用2025年数据"""
        print(f"🔍 历史预测验证: {stock_code}")
        print("=" * 60)
        
        # 使用默认验证时间（2025年）
        if start_date is None:
            start_date = self.default_time_config['validation_start']
        if end_date is None:
            end_date = self.default_time_config['validation_end']
        
        print(f"📅 验证时间范围: {start_date} 到 {end_date}")
        print(f"💡 这是模型训练时完全未使用的数据，用于验证模型效果")
        
        try:
            # 1. 加载模型
            model, scaler, model_info = self.load_model(stock_code)
            
            # 2. 加载特征数据
            feat = self.load_features(stock_code)
            
            # 3. 应用日期过滤
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
            
            # 过滤验证数据
            validation_data = feat[(feat.index >= start_date) & (feat.index <= end_date)].copy()
            
            if len(validation_data) == 0:
                raise ValueError(f"在指定时间范围内没有找到数据: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
            
            print(f"📈 验证数据: {len(validation_data)} 条记录")
            print(f"📅 实际验证范围: {validation_data.index.min().strftime('%Y-%m-%d')} 到 {validation_data.index.max().strftime('%Y-%m-%d')}")
            
            # 4. 准备验证数据
            feature_cols = model_info['feature_names']
            
            # 检查特征是否匹配
            missing_features = set(feature_cols) - set(validation_data.columns)
            if missing_features:
                raise ValueError(f"缺少特征: {missing_features}")
            
            X_new = validation_data[feature_cols].values
            
            # 移除包含NaN的行
            valid_mask = ~np.isnan(X_new).any(axis=1)
            X_new = X_new[valid_mask]
            validation_data_valid = validation_data[valid_mask].copy()
            
            print(f"🔍 有效验证数据: {len(validation_data_valid)} 条记录")
            print(f"📊 特征维度: {X_new.shape}")
            
            # 5. 标准化和预测
            print("🤖 进行历史验证...")
            X_new_scaled = scaler.transform(X_new)
            predictions = model.predict(X_new_scaled)
            probabilities = model.predict_proba(X_new_scaled)
            
            # 6. 添加预测结果到数据
            validation_data_valid['pred_label'] = predictions
            validation_data_valid['pred_prob'] = probabilities[:, 1]  # 上涨概率
            
            # 7. 分析验证结果
            self._analyze_validation_results(stock_code, validation_data_valid, model_info)
            
            # 8. 生成历史交易信号
            signals = self._generate_historical_signals(validation_data_valid)
            
            # 9. 保存验证结果
            self.prediction_history[stock_code] = {
                'predictions': validation_data_valid,
                'signals': signals,
                'model_info': model_info,
                'validation_date': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'validation_period': {
                    'start': start_date.strftime('%Y-%m-%d'),
                    'end': end_date.strftime('%Y-%m-%d'),
                    'samples': len(validation_data_valid)
                }
            }
            
            return validation_data_valid, signals
            
        except Exception as e:
            print(f"❌ 历史验证失败: {e}")
            return None, None
    
    def _analyze_validation_results(self, stock_code, validation_data, model_info):
        """分析历史验证结果"""
        print(f"\n📊 历史验证结果分析: {stock_code}")
        print("-" * 50)
        
        # 1. 基本统计
        total_samples = len(validation_data)
        avg_probability = np.mean(validation_data['pred_prob'])
        label_distribution = np.bincount(validation_data['pred_label'])
        
        print(f"📈 基本统计:")
        print(f"   验证样本数: {total_samples}")
        print(f"   平均预测概率: {avg_probability:.3f}")
        print(f"   预测上涨: {label_distribution[1]} ({label_distribution[1]/total_samples:.1%})")
        print(f"   预测下跌: {label_distribution[0]} ({label_distribution[0]/total_samples:.1%})")
        
        # 2. 概率分布分析
        high_confidence_up = np.sum(validation_data['pred_prob'] > 0.7)
        high_confidence_down = np.sum(validation_data['pred_prob'] < 0.3)
        medium_confidence = np.sum((validation_data['pred_prob'] >= 0.3) & (validation_data['pred_prob'] <= 0.7))
        
        print(f"\n🎯 置信度分析:")
        print(f"   高置信度上涨 (>0.7): {high_confidence_up} ({high_confidence_up/total_samples:.1%})")
        print(f"   高置信度下跌 (<0.3): {high_confidence_down} ({high_confidence_down/total_samples:.1%})")
        print(f"   中等置信度 (0.3-0.7): {medium_confidence} ({medium_confidence/total_samples:.1%})")
        
        # 3. 时间序列分析
        recent_predictions = validation_data.tail(20)  # 最近20个预测
        recent_avg_prob = np.mean(recent_predictions['pred_prob'])
        recent_trend = "上涨" if recent_avg_prob > avg_probability else "下跌"
        
        print(f"\n⏰ 近期趋势:")
        print(f"   最近20个预测平均概率: {recent_avg_prob:.3f}")
        print(f"   整体趋势: {recent_trend}")
        
        # 4. 与训练数据的对比
        if 'data_info' in model_info:
            training_info = model_info['data_info']
            print(f"\n📚 与训练数据对比:")
            print(f"   训练样本数: {training_info.get('training_samples', 'N/A')}")
            print(f"   测试样本数: {training_info.get('test_samples', 'N/A')}")
            print(f"   特征数量: {training_info.get('feature_count', 'N/A')}")
            print(f"   训练时间: {training_info.get('training_start', 'N/A')} 到 {training_info.get('training_end', 'N/A')}")
    
    def _generate_historical_signals(self, validation_data, buy_threshold=0.6, sell_threshold=0.4):
        """生成历史交易信号"""
        print(f"\n🚦 生成历史交易信号...")
        
        signals = []
        
        for i, row in validation_data.iterrows():
            prob = row['pred_prob']
            date = row.name
            
            if prob > buy_threshold:
                signal = {
                    'date': date,
                    'action': 'BUY',
                    'confidence': prob,
                    'reason': f'历史验证上涨概率: {prob:.3f} > {buy_threshold}'
                }
                signals.append(signal)
            elif prob < sell_threshold:
                signal = {
                    'date': date,
                    'action': 'SELL',
                    'confidence': 1 - prob,
                    'reason': f'历史验证下跌概率: {1-prob:.3f} > {1-sell_threshold}'
                }
                signals.append(signal)
            else:
                signal = {
                    'date': date,
                    'action': 'HOLD',
                    'confidence': 0.5,
                    'reason': f'历史验证概率: {prob:.3f}，建议观望'
                }
                signals.append(signal)
        
        # 统计信号
        buy_signals = [s for s in signals if s['action'] == 'BUY']
        sell_signals = [s for s in signals if s['action'] == 'SELL']
        hold_signals = [s for s in signals if s['action'] == 'HOLD']
        
        print(f"📊 历史交易信号统计:")
        print(f"   买入信号: {len(buy_signals)} 个")
        print(f"   卖出信号: {len(sell_signals)} 个")
        print(f"   观望信号: {len(hold_signals)} 个")
        
        return signals
    
    def save_validation_summary(self, filename=None):
        """保存历史验证结果摘要"""
        if not self.prediction_history:
            print("❌ 没有验证结果可保存")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"historical_validation_summary_{timestamp}.json"
        
        # 准备保存的数据
        save_data = {}
        for stock, data in self.prediction_history.items():
            save_data[stock] = {
                'validation_date': data['validation_date'],
                'validation_period': data['validation_period'],
                'model_info': {
                    'training_date': data['model_info'].get('training_date', 'Unknown'),
                    'feature_names': data['model_info'].get('feature_names', []),
                    'cv_scores': data['model_info'].get('cv_scores', {}),
                    'training_metrics': data['model_info'].get('training_metrics', {}),
                    'data_info': data['model_info'].get('data_info', {})
                },
                'validation_summary': {
                    'total_samples': len(data['predictions']),
                    'avg_probability': float(np.mean(data['predictions']['pred_prob'])),
                    'label_distribution': data['predictions']['pred_label'].value_counts().to_dict()
                },
                'signals_summary': {
                    'buy_signals': len([s for s in data['signals'] if s['action'] == 'BUY']),
                    'sell_signals': len([s for s in data['signals'] if s['action'] == 'SELL']),
                    'hold_signals': len([s for s in data['signals'] if s['action'] == 'HOLD'])
                }
            }
        
        # 保存到文件
        filepath = os.path.join(self.results_dir, 'reports', filename)
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"💾 历史验证摘要已保存: {filename}")
    
    def batch_validate_historical(self, stock_list=None, start_date=None, end_date=None):
        """批量验证多个股票的历史预测"""
        if stock_list is None:
            # 从特征目录获取股票列表
            if os.path.exists(self.features_dir):
                stock_list = [d for d in os.listdir(self.features_dir) if os.path.isdir(os.path.join(self.features_dir, d))]
            else:
                print("❌ 没有找到特征目录")
                return {}
        
        if not stock_list:
            print("❌ 没有股票可验证")
            return {}
        
        print(f"🔍 批量历史验证: {len(stock_list)} 只股票")
        print("=" * 60)
        print(f"📅 默认验证时间: {self.default_time_config['validation_start']} 到 {self.default_time_config['validation_end']}")
        print(f"💡 这是模型训练时完全未使用的数据，用于验证模型效果")
        print("=" * 60)
        
        results = {}
        
        for i, stock_code in enumerate(stock_list, 1):
            print(f"\n📊 [{i}/{len(stock_list)}] 验证股票: {stock_code}")
            print("-" * 40)
            
            try:
                feat_pred, signals = self.validate_historical_predictions(stock_code, start_date, end_date)
                
                if feat_pred is not None:
                    results[stock_code] = {
                        'success': True,
                        'samples': len(feat_pred),
                        'signals': len(signals),
                        'validation_period': self.prediction_history[stock_code]['validation_period']
                    }
                else:
                    results[stock_code] = {
                        'success': False,
                        'error': '验证失败'
                    }
                    
            except Exception as e:
                print(f"❌ {stock_code}: 验证异常 - {e}")
                results[stock_code] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
