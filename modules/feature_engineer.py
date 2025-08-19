#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程模块
构建技术指标特征，为机器学习模型提供输入
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """特征工程师"""
    
    def __init__(self, data_dir='data', features_dir='features'):
        self.data_dir = data_dir
        self.features_dir = features_dir
        self.ensure_directories()
        
    def ensure_directories(self):
        """确保必要的目录存在"""
        os.makedirs(self.features_dir, exist_ok=True)
    
    def load_cleaned_data(self, stock_code):
        """加载清洗后的数据"""
        cleaned_dir = os.path.join(self.data_dir, 'cleaned', stock_code)
        if not os.path.exists(cleaned_dir):
            print(f"❌ 没有找到清洗后的数据: {stock_code}")
            return None
        
        # 查找最新的数据文件
        import glob
        pattern = f"{stock_code}_*.csv"
        files = glob.glob(os.path.join(cleaned_dir, pattern))
        
        if not files:
            print(f"❌ 没有找到数据文件: {stock_code}")
            return None
        
        # 选择最新的文件
        latest_file = max(files, key=os.path.getctime)
        print(f"📂 加载数据: {os.path.basename(latest_file)}")
        
        try:
            df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
            print(f"✅ 数据加载成功: {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return None
    
    def build_features(self, df):
        """构建技术指标特征"""
        if df is None or df.empty:
            return None
        
        print(f"🔧 构建特征: {len(df)} 条记录")
        
        # 复制数据，避免修改原始数据
        feat = df.copy()
        
        # 1. 价格变化率
        feat['pct_change'] = feat['close'].pct_change()
        
        # 2. 移动平均线
        feat['ma5'] = feat['close'].rolling(window=5).mean()
        feat['ma10'] = feat['close'].rolling(window=10).mean()
        feat['ma20'] = feat['close'].rolling(window=20).mean()
        
        # 3. 价格与移动平均线的比率
        feat['ma5_ratio'] = feat['close'] / feat['ma5']
        feat['ma10_ratio'] = feat['close'] / feat['ma10']
        feat['ma20_ratio'] = feat['close'] / feat['ma20']
        
        # 4. MACD指标
        feat['ema12'] = feat['close'].ewm(span=12).mean()
        feat['ema26'] = feat['close'].ewm(span=26).mean()
        feat['macd_dif'] = feat['ema12'] - feat['ema26']
        feat['macd_dea'] = feat['macd_dif'].ewm(span=9).mean()
        feat['macd_hist'] = feat['macd_dif'] - feat['macd_dea']
        
        # 5. RSI指标
        feat['rsi14'] = self._calculate_rsi(feat['close'], window=14)
        
        # 6. 波动率指标
        feat['volatility_10'] = feat['pct_change'].rolling(window=10).std()
        feat['volatility_20'] = feat['pct_change'].rolling(window=20).std()
        
        # 7. 成交量指标
        feat['volume_ma5'] = feat['volume'].rolling(window=5).mean()
        feat['volume_ratio'] = feat['volume'] / feat['volume_ma5']
        
        # 8. 布林带
        feat['bb_upper'] = feat['ma20'] + 2 * feat['close'].rolling(window=20).std()
        feat['bb_lower'] = feat['ma20'] - 2 * feat['close'].rolling(window=20).std()
        feat['bb_position'] = (feat['close'] - feat['bb_lower']) / (feat['bb_upper'] - feat['bb_lower'])
        
        # 9. 动量指标
        feat['momentum_5'] = feat['close'] / feat['close'].shift(5) - 1
        feat['momentum_10'] = feat['close'] / feat['close'].shift(10) - 1
        
        # 10. 价格位置指标
        feat['high_low_ratio'] = (feat['high'] - feat['low']) / feat['close']
        feat['open_close_ratio'] = feat['open'] / feat['close']
        
        # 11. 趋势指标
        feat['trend_5'] = np.where(feat['ma5'] > feat['ma5'].shift(1), 1, -1)
        feat['trend_10'] = np.where(feat['ma10'] > feat['ma10'].shift(1), 1, -1)
        feat['trend_20'] = np.where(feat['ma20'] > feat['ma20'].shift(1), 1, -1)
        
        # 12. 交叉信号
        feat['ma5_cross_ma10'] = np.where(
            (feat['ma5'] > feat['ma10']) & (feat['ma5'].shift(1) <= feat['ma10'].shift(1)), 1, 0)
        feat['ma10_cross_ma20'] = np.where(
            (feat['ma10'] > feat['ma20']) & (feat['ma10'].shift(1) <= feat['ma20'].shift(1)), 1, 0)
        
        # 13. 支撑阻力位
        feat['support_level'] = feat['low'].rolling(window=20).min()
        feat['resistance_level'] = feat['high'].rolling(window=20).max()
        feat['support_distance'] = (feat['close'] - feat['support_level']) / feat['close']
        feat['resistance_distance'] = (feat['resistance_level'] - feat['close']) / feat['close']
        
        # 14. 价格通道
        feat['price_channel_high'] = feat['high'].rolling(window=20).max()
        feat['price_channel_low'] = feat['low'].rolling(window=20).min()
        feat['price_channel_position'] = (feat['close'] - feat['price_channel_low']) / (feat['price_channel_high'] - feat['price_channel_low'])
        
        # 15. 成交量价格关系
        feat['volume_price_trend'] = (feat['volume'] * feat['pct_change']).rolling(window=10).sum()
        
        # 16. 价格变化率特征（新增）
        feat['price_change'] = feat['close'].pct_change()
        feat['price_change_2'] = feat['close'].pct_change(2)
        feat['price_change_5'] = feat['close'].pct_change(5)
        
        # 17. 成交量变化特征（新增）
        feat['volume_change'] = feat['volume'].pct_change()
        feat['volume_ma_ratio'] = feat['volume'] / feat['volume'].rolling(5).mean()
        
        # 移除包含NaN的行
        initial_count = len(feat)
        feat = feat.dropna()
        print(f"   特征构建完成: {initial_count} -> {len(feat)} 条记录")
        
        # 添加元数据
        feat.attrs['stock_code'] = df.attrs.get('stock_code', 'unknown')
        feat.attrs['feature_count'] = len(feat.columns)
        feat.attrs['feature_names'] = list(feat.columns)
        feat.attrs['build_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return feat
    
    def _calculate_rsi(self, prices, window=14):
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def save_features(self, feat, stock_code):
        """保存特征数据"""
        if feat is None or feat.empty:
            print(f"❌ 没有特征数据可保存: {stock_code}")
            return False
        
        try:
            # 创建股票子目录
            stock_dir = os.path.join(self.features_dir, stock_code)
            os.makedirs(stock_dir, exist_ok=True)
            
            # 生成文件名
            start_date = feat.index.min().strftime('%Y%m%d')
            end_date = feat.index.max().strftime('%Y%m%d')
            filename = f"{stock_code}_features_{start_date}_{end_date}.csv"
            filepath = os.path.join(stock_dir, filename)
            
            # 保存CSV文件
            feat.to_csv(filepath, encoding='utf-8')
            
            # 保存特征信息
            feature_info = {
                'stock_code': stock_code,
                'feature_count': len(feat.columns),
                'feature_names': list(feat.columns),
                'data_points': len(feat),
                'date_range': f"{start_date} 到 {end_date}",
                'build_date': feat.attrs.get('build_date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                'file_path': filepath
            }
            
            info_file = os.path.join(stock_dir, f"{stock_code}_feature_info.json")
            import json
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(feature_info, f, ensure_ascii=False, indent=2)
            
            print(f"   💾 特征已保存: {filepath}")
            print(f"   📊 特征信息已保存: {info_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ 保存特征失败: {e}")
            return False
    
    def batch_build_features(self, stock_list=None, pool_name='all'):
        """批量构建特征"""
        if stock_list is None:
            # 从数据目录获取股票列表
            cleaned_dir = os.path.join(self.data_dir, 'cleaned')
            if os.path.exists(cleaned_dir):
                stock_list = [d for d in os.listdir(cleaned_dir) if os.path.isdir(os.path.join(cleaned_dir, d))]
            else:
                print("❌ 没有找到清洗后的数据目录")
                return {}
        
        if not stock_list:
            print("❌ 没有股票可处理")
            return {}
        
        print(f"🚀 开始批量构建特征: {len(stock_list)} 只股票")
        print("=" * 60)
        
        results = {}
        success_count = 0
        
        for i, stock_code in enumerate(stock_list, 1):
            print(f"\n🔧 [{i}/{len(stock_list)}] 处理股票: {stock_code}")
            print("-" * 40)
            
            try:
                # 加载数据
                df = self.load_cleaned_data(stock_code)
                
                if df is not None and not df.empty:
                    # 构建特征
                    feat = self.build_features(df)
                    
                    if feat is not None and not feat.empty:
                        # 保存特征
                        if self.save_features(feat, stock_code):
                            results[stock_code] = {
                                'status': 'success',
                                'feature_count': len(feat.columns),
                                'data_points': len(feat),
                                'date_range': f"{feat.index.min().strftime('%Y-%m-%d')} 到 {feat.index.max().strftime('%Y-%m-%d')}"
                            }
                            success_count += 1
                        else:
                            results[stock_code] = {'status': 'save_failed'}
                    else:
                        results[stock_code] = {'status': 'feature_build_failed'}
                else:
                    results[stock_code] = {'status': 'data_load_failed'}
                    
            except Exception as e:
                print(f"   ❌ 处理失败: {e}")
                results[stock_code] = {'status': 'error', 'error': str(e)}
            
            print(f"   {'✅' if results[stock_code]['status'] == 'success' else '❌'} {stock_code}")
        
        # 生成特征构建报告
        self._generate_feature_report(results)
        
        print(f"\n🎉 批量特征构建完成!")
        print(f"   成功: {success_count}/{len(stock_list)}")
        print(f"   失败: {len(stock_list) - success_count}/{len(stock_list)}")
        
        return results
    
    def _generate_feature_report(self, results):
        """生成特征构建报告"""
        try:
            report = {
                'build_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_stocks': len(results),
                'success_count': sum(1 for r in results.values() if r['status'] == 'success'),
                'failed_count': len(results) - sum(1 for r in results.values() if r['status'] == 'success'),
                'results': results
            }
            
            # 保存报告
            report_file = os.path.join(self.features_dir, f'feature_build_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            import json
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"\n📋 特征构建报告已保存: {report_file}")
            
        except Exception as e:
            print(f"❌ 生成报告失败: {e}")
    
    def get_feature_summary(self):
        """获取特征摘要"""
        print("📊 特征摘要")
        print("=" * 60)
        
        if not os.path.exists(self.features_dir):
            print("❌ 特征目录不存在")
            return
        
        stock_dirs = [d for d in os.listdir(self.features_dir) if os.path.isdir(os.path.join(self.features_dir, d))]
        
        if not stock_dirs:
            print("❌ 没有特征数据")
            return
        
        print(f"📁 特征数据 ({len(stock_dirs)} 只股票):")
        
        for stock_dir in sorted(stock_dirs):
            stock_path = os.path.join(self.features_dir, stock_dir)
            files = [f for f in os.listdir(stock_path) if f.endswith('.csv')]
            
            if files:
                # 获取最新文件的信息
                latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(stock_path, x)))
                file_path = os.path.join(stock_path, latest_file)
                
                try:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    print(f"   {stock_dir}: {len(df.columns)} 个特征, {len(df)} 条记录")
                    print(f"      时间范围: {df.index.min().strftime('%Y-%m-%d')} 到 {df.index.max().strftime('%Y-%m-%d')}")
                except:
                    print(f"   {stock_dir}: 文件读取失败")

def main():
    """主函数"""
    print("🔧 特征工程模块")
    print("=" * 60)
    
    # 创建特征工程师
    fe = FeatureEngineer()
    
    print("🔧 系统状态:")
    print(f"   数据目录: {os.path.abspath(fe.data_dir)}")
    print(f"   特征目录: {os.path.abspath(fe.features_dir)}")
    
    # 检查是否有清洗后的数据
    cleaned_dir = os.path.join(fe.data_dir, 'cleaned')
    if not os.path.exists(cleaned_dir):
        print("\n❌ 没有找到清洗后的数据目录!")
        print("   请先运行数据获取模块下载数据")
        return
    
    # 获取可用的股票
    stock_dirs = [d for d in os.listdir(cleaned_dir) if os.path.isdir(os.path.join(cleaned_dir, d))]
    
    if not stock_dirs:
        print("\n❌ 没有找到股票数据!")
        print("   请先运行数据获取模块下载数据")
        return
    
    print(f"\n📊 发现 {len(stock_dirs)} 只股票的数据:")
    for stock_dir in stock_dirs:
        print(f"   • {stock_dir}")
    
    # 开始构建特征
    print(f"\n🚀 开始构建特征...")
    results = fe.batch_build_features(stock_list=stock_dirs)
    
    # 显示特征摘要
    print(f"\n📊 特征构建完成，查看摘要:")
    fe.get_feature_summary()

if __name__ == "__main__":
    main()
