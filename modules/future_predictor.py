# -*- coding: utf-8 -*-
"""
未来预测器
真正的未来预测，基于最新特征预测未来1-5天的涨跌
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
        self.max_prediction_days = 5  # 最大预测天数限制
        
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
    
    def predict_future_dates(self, stock_code, future_dates, confidence_threshold=0.6):
        """预测指定未来日期的涨跌"""
        print(f"🔮 未来预测: {stock_code}")
        print("=" * 60)
        
        # 检查预测天数限制
        if len(future_dates) > self.max_prediction_days:
            print(f"⚠️  请求预测 {len(future_dates)} 天，超过最大限制 {self.max_prediction_days} 天")
            print(f"   将自动调整为预测前 {self.max_prediction_days} 天")
            future_dates = future_dates[:self.max_prediction_days]
        
        print(f"📅 预测日期: {len(future_dates)} 天")
        for i, date in enumerate(future_dates, 1):
            print(f"   {i}. {date}")
        
        try:
            # 1. 加载模型
            model, scaler, model_info = self.load_model(stock_code)
            
            # 2. 加载特征数据
            feat = self.load_features(stock_code)
            
            # 3. 获取最新特征（用于预测未来）
            latest_features = feat.iloc[-1:].copy()  # 最新一天的特征
            
            # 4. 准备预测数据
            feature_cols = model_info['feature_names']
            
            # 检查特征是否匹配
            missing_features = set(feature_cols) - set(latest_features.columns)
            if missing_features:
                raise ValueError(f"缺少特征: {missing_features}")
            
            X_new = latest_features[feature_cols].values
            
            # 检查数据有效性
            if np.isnan(X_new).any():
                raise ValueError("最新特征数据包含NaN值，无法进行预测")
            
            # 5. 标准化和预测
            print("🤖 进行未来预测...")
            X_new_scaled = scaler.transform(X_new)
            predictions = model.predict(X_new_scaled)
            probabilities = model.predict_proba(X_new_scaled)
            
            # 6. 生成预测结果
            prediction_results = []
            
            for i, future_date in enumerate(future_dates):
                # 检查日期是否在未来
                future_dt = pd.to_datetime(future_date)
                today = pd.Timestamp.now().normalize()
                
                if future_dt <= today:
                    print(f"⚠️  {future_date}: 不是未来日期，跳过")
                    continue
                
                # 使用最新特征预测
                pred_label = predictions[0]
                pred_prob = probabilities[0, 1]  # 上涨概率
                
                # 生成交易信号
                if pred_prob > confidence_threshold:
                    action = "BUY"
                    emoji = "📈"
                    signal_strength = "强"
                elif pred_prob < (1 - confidence_threshold):
                    action = "SELL"
                    emoji = "📉"
                    signal_strength = "强"
                else:
                    action = "HOLD"
                    emoji = "⏸️"
                    signal_strength = "弱"
                
                prediction_result = {
                    'date': future_date,
                    'prediction': int(pred_label),  # 转换为Python int
                    'probability': float(pred_prob),  # 转换为Python float
                    'action': action,
                    'signal_strength': signal_strength,
                    'emoji': emoji,
                    'confidence': float(pred_prob if action == "BUY" else (1 - pred_prob))  # 转换为Python float
                }
                
                prediction_results.append(prediction_result)
                
                # 打印每日预测结果
                print(f"   📅 {future_date}: {emoji} {action} - 概率: {pred_prob:.3f} ({signal_strength})")
            
            if not prediction_results:
                print("❌ 没有有效的未来日期可预测")
                return None
            
            # 7. 生成预测摘要
            self._generate_prediction_summary(stock_code, prediction_results)
            
            # 8. 保存预测结果
            self._save_future_predictions(stock_code, prediction_results)
            
            return {
                'stock_code': stock_code,
                'total_predictions': len(prediction_results),
                'predictions': prediction_results,
                'prediction_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            print(f"❌ 未来预测失败: {e}")
            return None
    
    def predict_next_n_days(self, stock_code, n_days=5, confidence_threshold=0.6):
        """预测未来n天的涨跌"""
        print(f"🔮 预测未来 {n_days} 天: {stock_code}")
        
        # 限制预测天数
        if n_days > self.max_prediction_days:
            print(f"⚠️  请求预测 {n_days} 天，超过最大限制 {self.max_prediction_days} 天")
            print(f"   将自动调整为预测 {self.max_prediction_days} 天")
            n_days = self.max_prediction_days
        
        # 生成未来日期
        future_dates = self._get_future_trading_days(n_days)
        
        return self.predict_future_dates(stock_code, future_dates, confidence_threshold)
    
    def _get_future_trading_days(self, n_days):
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
        print(f"\n📊 未来预测摘要: {stock_code}")
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
    
    def batch_predict_future(self, stock_list, future_dates, confidence_threshold=0.6):
        """批量预测多个股票的未来涨跌"""
        if not stock_list:
            print("❌ 没有股票可预测")
            return {}
        
        # 限制预测天数
        if len(future_dates) > self.max_prediction_days:
            print(f"⚠️  请求预测 {len(future_dates)} 天，超过最大限制 {self.max_prediction_days} 天")
            print(f"   将自动调整为预测前 {self.max_prediction_days} 天")
            future_dates = future_dates[:self.max_prediction_days]
        
        print(f"🔮 批量未来预测: {len(stock_list)} 只股票，{len(future_dates)} 天")
        print("=" * 60)
        
        results = {}
        
        for i, stock_code in enumerate(stock_list, 1):
            print(f"\n📊 [{i}/{len(stock_list)}] 预测股票: {stock_code}")
            print("-" * 40)
            
            try:
                result = self.predict_future_dates(stock_code, future_dates, confidence_threshold)
                
                if result:
                    results[stock_code] = {
                        'success': True,
                        'predictions': result['predictions'],
                        'total_predictions': result['total_predictions']
                    }
                else:
                    results[stock_code] = {
                        'success': False,
                        'error': '预测失败'
                    }
                    
            except Exception as e:
                print(f"❌ {stock_code}: 预测异常 - {e}")
                results[stock_code] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def get_prediction_limits(self):
        """获取预测限制信息"""
        return {
            'max_prediction_days': self.max_prediction_days,
            'description': f"最多只能预测未来 {self.max_prediction_days} 天的涨跌"
        }
