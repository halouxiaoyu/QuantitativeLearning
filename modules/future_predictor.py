# -*- coding: utf-8 -*-
"""
未来预测器
真正的未来预测，基于最新特征预测未来1-2天的涨跌
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class FuturePredictor:
    """未来预测器 - 预测未来日期的股票涨跌"""
    
    def __init__(self, features_dir, models_dir, results_dir):
        self.features_dir = features_dir
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.max_prediction_days = 5  # 最大预测天数限制为5天
        
        # 创建必要的目录
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'future_predictions'), exist_ok=True)
    
    def load_model(self, stock_code):
        """加载训练好的模型"""
        model_dir = os.path.join(self.models_dir, stock_code)
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"未找到模型目录: {stock_code}")
        
        # 查找最新的模型文件
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and 'model' in f]
        if not model_files:
            raise FileNotFoundError(f"未找到模型文件: {stock_code}")
        
        # 加载模型
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(model_dir, latest_model)
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
    
    def predict_next_n_days(self, stock_code, n_days=2, confidence_threshold=0.6):
        """预测未来n天的涨跌 - 使用滚动预测方法"""
        print(f"🔮 预测未来{n_days}天: {stock_code}")
        print("=" * 60)
        
        # 限制预测天数
        if n_days > self.max_prediction_days:
            print(f"⚠️  请求预测 {n_days} 天，超过最大限制 {self.max_prediction_days} 天")
            print(f"   将自动调整为预测 {self.max_prediction_days} 天")
            n_days = self.max_prediction_days
        
        try:
            # 1. 加载模型和特征
            model, scaler, model_info = self.load_model(stock_code)
            feat = self.load_features(stock_code)
            
            # 2. 获取最新特征（用于预测明天）
            latest_features = feat.iloc[-1:].copy()
            feature_cols = model_info['feature_names']
            
            # 检查特征是否匹配
            missing_features = set(feature_cols) - set(latest_features.columns)
            if missing_features:
                raise ValueError(f"缺少特征: {missing_features}")
            
            # 3. 预测明天（第1天）
            print("🤖 预测明天（第1天）...")
            X_tomorrow = latest_features[feature_cols].values
            
            if np.isnan(X_tomorrow).any():
                raise ValueError("最新特征数据包含NaN值，无法进行预测")
            
            X_tomorrow_scaled = scaler.transform(X_tomorrow)
            tomorrow_pred = model.predict(X_tomorrow_scaled)[0]
            tomorrow_probs = model.predict_proba(X_tomorrow_scaled)[0]
            tomorrow_prob_up = tomorrow_probs[1]  # 上涨概率
            tomorrow_prob_down = tomorrow_probs[0]  # 下跌概率
            
            # 根据概率确定预测标签
            if tomorrow_prob_up > tomorrow_prob_down:
                tomorrow_pred = 1  # 上涨
                tomorrow_prob = tomorrow_prob_up
            else:
                tomorrow_pred = 0  # 下跌
                tomorrow_prob = tomorrow_prob_down
            
            # 4. 基于明天的预测结果，预测后天（第2天）
            print("🤖 预测后天（第2天）...")
            
            # 创建后天的特征（基于明天的预测结果调整）
            X_day_after = self._create_next_day_features(feat, tomorrow_pred, tomorrow_prob, feature_cols)
            X_day_after_scaled = scaler.transform(X_day_after)
            day_after_pred = model.predict(X_day_after_scaled)[0]
            day_after_probs = model.predict_proba(X_day_after_scaled)[0]
            day_after_prob_up = day_after_probs[1]  # 上涨概率
            day_after_prob_down = day_after_probs[0]  # 下跌概率
            
            # 根据概率确定预测标签
            if day_after_prob_up > day_after_prob_down:
                day_after_pred = 1  # 上涨
                day_after_prob = day_after_prob_up
            else:
                day_after_pred = 0  # 下跌
                day_after_prob = day_after_prob_down
            
            # 5. 生成未来n天的预测结果
            future_dates = self._get_next_n_trading_days(n_days)
            prediction_results = []
            
            # 第一天的预测（明天）
            first_day_result = self._create_prediction_result(
                future_dates[0], tomorrow_pred, tomorrow_prob, confidence_threshold
            )
            prediction_results.append(first_day_result)
            
            # 后续天的预测（基于前一天的预测结果）
            current_features = feat.copy()
            current_pred = tomorrow_pred
            current_prob = tomorrow_prob
            
            for i in range(1, n_days):
                # 基于前一天的预测结果，创建下一天的特征
                next_day_features = self._create_next_day_features(current_features, current_pred, current_prob, feature_cols)
                next_day_scaled = scaler.transform(next_day_features)
                next_day_pred = model.predict(next_day_scaled)[0]
                next_day_probs = model.predict_proba(next_day_scaled)[0]
                next_day_prob_up = next_day_probs[1]  # 上涨概率
                next_day_prob_down = next_day_probs[0]  # 下跌概率
                
                # 根据概率确定预测标签
                if next_day_prob_up > next_day_prob_down:
                    next_day_pred = 1  # 上涨
                    next_day_prob = next_day_prob_up
                else:
                    next_day_pred = 0  # 下跌
                    next_day_prob = next_day_prob_down
                
                # 创建预测结果
                next_day_result = self._create_prediction_result(
                    future_dates[i], next_day_pred, next_day_prob, confidence_threshold
                )
                prediction_results.append(next_day_result)
                
                # 更新当前状态，用于下一次预测
                current_features = self._update_features_for_next_day(current_features, next_day_pred, next_day_prob)
                current_pred = next_day_pred
                current_prob = next_day_prob
            
            # 6. 生成预测摘要
            self._generate_prediction_summary(stock_code, prediction_results)
            
            # 7. 保存预测结果
            self._save_future_predictions(stock_code, prediction_results)
            
            return {
                'stock_code': stock_code,
                'total_predictions': len(prediction_results),
                'predictions': prediction_results,
                'prediction_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'method': f'滚动预测（基于前一天结果预测下一天，共{n_days}天）'
            }
            
        except Exception as e:
            print(f"❌ 未来预测失败: {e}")
            return None
    
    def _create_next_day_features(self, feat, prev_pred, prev_prob, feature_cols):
        """基于前一天的预测结果，创建下一天的特征"""
        # 获取最新特征作为基础
        latest_features = feat.iloc[-1:].copy()
        
        # 创建新的特征行
        next_day_features = latest_features.copy()
        
        # 根据前一天的预测结果调整特征
        if prev_pred == 1:  # 前一天预测上涨
            # 调整价格相关特征，模拟上涨趋势
            price_features = ['close', 'high', 'low', 'open']
            for col in price_features:
                if col in next_day_features.columns:
                    # 小幅上涨（1-3%）
                    increase = 1 + (0.01 + prev_prob * 0.02)  # 基于概率调整涨幅
                    next_day_features[col] = next_day_features[col] * increase
            
            # 调整技术指标特征
            if 'rsi_14' in next_day_features.columns:
                next_day_features['rsi_14'] = min(70, next_day_features['rsi_14'] + 5)  # RSI上升
            
            if 'macd' in next_day_features.columns:
                next_day_features['macd'] = next_day_features['macd'] * 1.1  # MACD增强
            
        else:  # 前一天预测下跌
            # 调整价格相关特征，模拟下跌趋势
            price_features = ['close', 'high', 'low', 'open']
            for col in price_features:
                if col in next_day_features.columns:
                    # 小幅下跌（1-3%）
                    decrease = 1 - (0.01 + (1 - prev_prob) * 0.02)  # 基于概率调整跌幅
                    next_day_features[col] = next_day_features[col] * decrease
            
            # 调整技术指标特征
            if 'rsi_14' in next_day_features.columns:
                next_day_features['rsi_14'] = max(30, next_day_features['rsi_14'] - 5)  # RSI下降
            
            if 'macd' in next_day_features.columns:
                next_day_features['macd'] = next_day_features['macd'] * 0.9  # MACD减弱
        
        # 重新计算衍生特征
        next_day_features = self._recalculate_derived_features(next_day_features, feature_cols)
        
        return next_day_features[feature_cols].values
    
    def _update_features_for_next_day(self, features, prev_pred, prev_prob):
        """更新特征，为下一次预测做准备"""
        # 基于预测结果更新特征
        if prev_pred == 1:  # 上涨
            # 更新价格特征
            price_features = ['close', 'high', 'low', 'open']
            for col in price_features:
                if col in features.columns:
                    increase = 1 + (0.01 + prev_prob * 0.02)
                    features[col] = features[col] * increase
        else:  # 下跌
            price_features = ['close', 'high', 'low', 'open']
            for col in price_features:
                if col in features.columns:
                    decrease = 1 - (0.01 + (1 - prev_prob) * 0.02)
                    features[col] = features[col] * decrease
        
        return features
    
    def _recalculate_derived_features(self, features, feature_cols):
        """重新计算衍生特征"""
        # 这里可以添加一些简单的特征重新计算逻辑
        # 比如移动平均、动量等
        return features
    
    def _create_prediction_result(self, date, pred_label, pred_prob, confidence_threshold):
        """创建预测结果"""
        # 生成交易信号
        if pred_label == 1:  # 预测上涨
            if pred_prob > confidence_threshold:
                action = "BUY"
                emoji = "📈"
                signal_strength = "强"
            else:
                action = "HOLD"
                emoji = "⏸️"
                signal_strength = "弱"
        else:  # 预测下跌
            if pred_prob > confidence_threshold:
                action = "SELL"
                emoji = "📉"
                signal_strength = "强"
            else:
                action = "HOLD"
                emoji = "⏸️"
                signal_strength = "弱"
        
        return {
            'date': date,
            'prediction': int(pred_label),
            'probability': float(pred_prob),
            'action': action,
            'signal_strength': signal_strength,
            'emoji': emoji,
            'confidence': float(pred_prob)  # 置信度就是预测结果的概率
        }
    
    def _get_next_n_trading_days(self, n_days):
        """获取未来n个交易日"""
        today = datetime.now()
        future_dates = []
        current_date = today
        
        while len(future_dates) < n_days:
            current_date += timedelta(days=1)
            
            # 跳过周末
            if current_date.weekday() < 5:  # 0-4 是周一到周五
                future_dates.append(current_date.strftime('%Y-%m-%d'))
        
        return future_dates
    
    def _generate_prediction_summary(self, stock_code, predictions):
        """生成预测摘要"""
        print(f"\n📊 未来2天预测摘要: {stock_code}")
        print("-" * 50)
        
        # 统计信号
        buy_count = len([p for p in predictions if p['action'] == 'BUY'])
        sell_count = len([p for p in predictions if p['action'] == 'SELL'])
        hold_count = len([p for p in predictions if p['action'] == 'HOLD'])
        
        print(f"🚦 交易信号统计:")
        print(f"   买入信号: {buy_count} 天")
        print(f"   卖出信号: {sell_count} 天")
        print(f"   观望信号: {hold_count} 天")
        
        # 平均置信度
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        print(f"🎯 平均置信度: {avg_confidence:.3f}")
        
        # 高置信度预测
        high_confidence = [p for p in predictions if p['confidence'] > 0.7]
        print(f"⭐ 高置信度预测: {len(high_confidence)} 天")
        
        # 每日预测结果汇总
        print(f"\n📅 每日预测结果汇总:")
        for pred in predictions:
            print(f"   {pred['date']}: {pred['emoji']} {pred['action']} - 概率: {pred['probability']:.3f} ({pred['signal_strength']})")
    
    def _save_future_predictions(self, stock_code, predictions):
        """保存未来预测结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{stock_code}_future_predictions_{timestamp}.csv"
            filepath = os.path.join(self.results_dir, 'future_predictions', filename)
            
            # 转换为DataFrame并保存
            df = pd.DataFrame(predictions)
            df.to_csv(filepath, index=False, encoding='utf-8')
            
            print(f"💾 未来预测结果已保存: {filename}")
            
        except Exception as e:
            print(f"❌ 保存预测结果失败: {e}")
    
    def predict_next_2_days(self, stock_code, confidence_threshold=0.6):
        """预测未来2天的涨跌（兼容性方法）"""
        return self.predict_next_n_days(stock_code, n_days=2, confidence_threshold=confidence_threshold)
    
    def get_prediction_limits(self):
        """获取预测限制信息"""
        return {
            'max_prediction_days': self.max_prediction_days,
            'description': f"系统现在支持预测未来 {self.max_prediction_days} 天的涨跌，使用滚动预测方法"
        }
    
    def get_available_stocks(self):
        """获取可预测的股票列表"""
        available_stocks = []
        
        if os.path.exists(self.features_dir):
            for stock_dir in os.listdir(self.features_dir):
                if os.path.isdir(os.path.join(self.features_dir, stock_dir)):
                    # 检查是否有特征文件
                    stock_features_dir = os.path.join(self.features_dir, stock_dir)
                    feature_files = [f for f in os.listdir(stock_features_dir) if f.endswith('.csv') and 'features' in f]
                    
                    # 检查是否有模型文件
                    stock_models_dir = os.path.join(self.models_dir, stock_dir)
                    model_files = []
                    if os.path.exists(stock_models_dir):
                        model_files = [f for f in os.listdir(stock_models_dir) if f.endswith('.pkl') and 'model' in f]
                    
                    if feature_files and model_files:
                        # 获取最新特征文件信息
                        latest_feature = sorted(feature_files)[-1]
                        feature_path = os.path.join(stock_features_dir, latest_feature)
                        
                        try:
                            df = pd.read_csv(feature_path, index_col=0, parse_dates=True)
                            available_stocks.append({
                                'stock_code': stock_dir,
                                'features_count': len(df.columns),
                                'records_count': len(df),
                                'latest_date': df.index.max().strftime('%Y-%m-%d'),
                                'feature_file': latest_feature
                            })
                        except:
                            continue
        
        return available_stocks
    
    def check_stock_status(self, stock_code):
        """检查股票状态"""
        status = {
            'stock_code': stock_code,
            'has_features': False,
            'has_model': False,
            'can_predict': False,
            'missing_items': []
        }
        
        # 检查特征
        stock_features_dir = os.path.join(self.features_dir, stock_code)
        if os.path.exists(stock_features_dir):
            feature_files = [f for f in os.listdir(stock_features_dir) if f.endswith('.csv') and 'features' in f]
            if feature_files:
                status['has_features'] = True
            else:
                status['missing_items'].append('特征文件')
        else:
            status['missing_items'].append('特征目录')
        
        # 检查模型
        stock_models_dir = os.path.join(self.models_dir, stock_code)
        if os.path.exists(stock_models_dir):
            model_files = [f for f in os.listdir(stock_models_dir) if f.endswith('.pkl') and 'model' in f]
            if model_files:
                status['has_model'] = True
            else:
                status['missing_items'].append('训练好的模型')
        else:
            status['missing_items'].append('模型目录')
        
        status['can_predict'] = status['has_features'] and status['has_model']
        
        return status
