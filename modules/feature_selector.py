#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征选择模块
帮助识别和选择最重要的特征，减少噪音，提高模型性能
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 机器学习相关
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureSelector:
    """特征选择器"""
    
    def __init__(self, features_dir='features', results_dir='results'):
        self.features_dir = features_dir
        self.results_dir = results_dir
        self.ensure_directories()
        
    def ensure_directories(self):
        """确保必要的目录存在"""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'feature_analysis'), exist_ok=True)
    
    def load_features(self, stock_code):
        """加载特征数据"""
        stock_dir = os.path.join(self.features_dir, stock_code)
        if not os.path.exists(stock_dir):
            print(f"❌ 没有找到特征数据: {stock_code}")
            return None
        
        # 查找最新的特征文件
        import glob
        pattern = f"{stock_code}_features_*.csv"
        files = glob.glob(os.path.join(stock_dir, pattern))
        
        if not files:
            print(f"❌ 没有找到特征文件: {stock_code}")
            return None
        
        # 选择最新的文件
        latest_file = max(files, key=os.path.getctime)
        print(f"📂 加载特征: {os.path.basename(latest_file)}")
        
        try:
            feat = pd.read_csv(latest_file, index_col=0, parse_dates=True)
            print(f"✅ 特征加载成功: {len(feat)} 条记录")
            return feat
        except Exception as e:
            print(f"❌ 特征加载失败: {e}")
            return None
    
    def prepare_data(self, feat, start_date='2021-01-01', end_date='2024-12-31'):
        """准备训练数据"""
        print(f"🔧 准备训练数据: {start_date} 到 {end_date}")
        
        # 过滤时间范围
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        training_data = feat[(feat.index >= start_dt) & (feat.index <= end_dt)].copy()
        
        if len(training_data) == 0:
            print(f"❌ 在指定时间范围内没有找到数据")
            return None, None, None
        
        # 准备特征和标签
        feature_cols = [col for col in training_data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        if len(feature_cols) == 0:
            print(f"❌ 没有找到有效的特征列")
            return None, None, None
        
        X = training_data[feature_cols].values
        y = (training_data['close'].shift(-1) > training_data['close']).astype(int)
        
        # 移除最后一行（因为没有下一天的价格）
        X = X[:-1]
        y = y[:-1]
        
        # 移除包含NaN的行
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"🔍 有效训练样本: {len(X)} 条")
        print(f"📊 特征维度: {X.shape}")
        print(f"🎯 标签分布: 上涨 {np.sum(y)} ({np.sum(y)/len(y):.1%}), 下跌 {np.sum(~y)} ({np.sum(~y)/len(y):.1%})")
        
        return X, y, feature_cols
    
    def analyze_feature_importance(self, stock_code, start_date='2021-01-01', end_date='2024-12-31', n_features=20):
        """分析特征重要性"""
        print(f"🔍 分析特征重要性: {stock_code}")
        print("=" * 60)
        
        # 加载数据
        feat = self.load_features(stock_code)
        if feat is None:
            return None
        
        # 准备训练数据
        X, y, feature_cols = self.prepare_data(feat, start_date, end_date)
        if X is None:
            return None
        
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 1. 随机森林特征重要性
        print("🌲 使用随机森林分析特征重要性...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y)
        
        # 获取特征重要性
        feature_importance = rf.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"📊 随机森林特征重要性排序 (前10名):")
        for i, row in feature_importance_df.head(10).iterrows():
            print(f"   {row['feature']:25} : {row['importance']:.4f}")
        
        # 2. 统计检验特征选择
        print("\n📈 使用统计检验选择特征...")
        selector_f = SelectKBest(score_func=f_classif, k=n_features)
        X_selected_f = selector_f.fit_transform(X_scaled, y)
        selected_features_f = [feature_cols[i] for i in selector_f.get_support(indices=True)]
        
        print(f"📊 F检验选择的特征 ({len(selected_features_f)} 个):")
        for i, feature in enumerate(selected_features_f, 1):
            print(f"   {i:2d}. {feature}")
        
        # 3. 互信息特征选择
        print("\n📊 使用互信息选择特征...")
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=n_features)
        X_selected_mi = selector_mi.fit_transform(X_scaled, y)
        selected_features_mi = [feature_cols[i] for i in selector_mi.get_support(indices=True)]
        
        print(f"📊 互信息选择的特征 ({len(selected_features_mi)} 个):")
        for i, feature in enumerate(selected_features_mi, 1):
            print(f"   {i:2d}. {feature}")
        
        # 4. 交叉验证性能对比
        print("\n🧪 交叉验证性能对比...")
        
        # 原始特征
        cv_scores_original = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy')
        print(f"📊 原始特征 (42个): {cv_scores_original.mean():.3f} ± {cv_scores_original.std() * 2:.3f}")
        
        # 随机森林选择的特征
        top_features_rf = feature_importance_df.head(n_features)['feature'].tolist()
        X_rf_selected = X[:, [feature_cols.index(f) for f in top_features_rf]]
        X_rf_scaled = scaler.fit_transform(X_rf_selected)
        cv_scores_rf = cross_val_score(rf, X_rf_scaled, y, cv=5, scoring='accuracy')
        print(f"📊 随机森林选择 ({n_features}个): {cv_scores_rf.mean():.3f} ± {cv_scores_rf.std() * 2:.3f}")
        
        # F检验选择的特征
        X_f_scaled = scaler.fit_transform(X_selected_f)
        cv_scores_f = cross_val_score(rf, X_f_scaled, y, cv=5, scoring='accuracy')
        print(f"📊 F检验选择 ({n_features}个): {cv_scores_f.mean():.3f} ± {cv_scores_f.std() * 2:.3f}")
        
        # 5. 生成特征分析报告
        analysis_result = {
            'stock_code': stock_code,
            'analysis_date': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'original_features': len(feature_cols),
            'selected_features': n_features,
            'feature_importance': feature_importance_df.to_dict('records'),
            'rf_selected_features': top_features_rf,
            'f_test_selected_features': selected_features_f,
            'mi_selected_features': selected_features_mi,
            'cv_performance': {
                'original': {
                    'mean': float(cv_scores_original.mean()),
                    'std': float(cv_scores_original.std()),
                    'scores': cv_scores_original.tolist()
                },
                'rf_selected': {
                    'mean': float(cv_scores_rf.mean()),
                    'std': float(cv_scores_rf.std()),
                    'scores': cv_scores_rf.tolist()
                },
                'f_test_selected': {
                    'mean': float(cv_scores_f.mean()),
                    'std': float(cv_scores_f.std()),
                    'scores': cv_scores_f.tolist()
                }
            }
        }
        
        # 保存分析结果
        self.save_analysis_result(analysis_result)
        
        # 生成可视化图表
        self.generate_feature_plots(feature_importance_df, analysis_result)
        
        return analysis_result
    
    def save_analysis_result(self, analysis_result):
        """保存特征分析结果"""
        try:
            stock_code = analysis_result['stock_code']
            timestamp = analysis_result['analysis_date']
            
            # 创建股票子目录
            stock_dir = os.path.join(self.results_dir, 'feature_analysis', stock_code)
            os.makedirs(stock_dir, exist_ok=True)
            
            # 保存分析结果
            result_file = os.path.join(stock_dir, f"{stock_code}_feature_analysis_{timestamp}.json")
            import json
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, ensure_ascii=False, indent=2)
            
            print(f"💾 特征分析结果已保存: {os.path.basename(result_file)}")
            
        except Exception as e:
            print(f"❌ 保存特征分析结果失败: {e}")
    
    def generate_feature_plots(self, feature_importance_df, analysis_result):
        """生成特征重要性图表"""
        try:
            stock_code = analysis_result['stock_code']
            timestamp = analysis_result['analysis_date']
            
            # 创建图表目录
            plots_dir = os.path.join(self.results_dir, 'feature_analysis', stock_code, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 1. 特征重要性条形图
            plt.figure(figsize=(12, 8))
            top_features = feature_importance_df.head(20)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('特征重要性')
            plt.title(f'{stock_code} - 特征重要性排序 (前20名)')
            plt.tight_layout()
            
            plot_file = os.path.join(plots_dir, f"{stock_code}_feature_importance_{timestamp}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 特征重要性图表已保存: {os.path.basename(plot_file)}")
            
        except Exception as e:
            print(f"❌ 生成特征图表失败: {e}")
    
    def get_optimal_feature_set(self, stock_code, start_date='2021-01-01', end_date='2024-12-31'):
        """获取最优特征集"""
        print(f"🎯 寻找最优特征集: {stock_code}")
        print("=" * 60)
        
        # 分析特征重要性
        analysis_result = self.analyze_feature_importance(stock_code, start_date, end_date)
        if analysis_result is None:
            return None
        
        # 根据交叉验证性能选择最佳特征集
        cv_performance = analysis_result['cv_performance']
        
        best_method = None
        best_score = 0
        
        for method, scores in cv_performance.items():
            if scores['mean'] > best_score:
                best_score = scores['mean']
                best_method = method
        
        print(f"\n🏆 最佳特征选择方法: {best_method}")
        print(f"📊 最佳交叉验证准确率: {best_score:.3f}")
        
        if best_method == 'rf_selected':
            optimal_features = analysis_result['rf_selected_features']
        elif best_method == 'f_test_selected':
            optimal_features = analysis_result['f_test_selected_features']
        else:
            optimal_features = analysis_result['rf_selected_features']  # 默认使用随机森林选择
        
        print(f"🎯 推荐特征数量: {len(optimal_features)}")
        print(f"📋 推荐特征列表:")
        for i, feature in enumerate(optimal_features, 1):
            print(f"   {i:2d}. {feature}")
        
        return optimal_features

def main():
    """主函数"""
    print("🔍 特征选择模块")
    print("=" * 60)
    
    fs = FeatureSelector()
    
    # 检查是否有特征数据
    if not os.path.exists(fs.features_dir):
        print("❌ 没有找到特征目录!")
        print("   请先运行特征工程模块构建特征")
        return
    
    # 获取可用的股票
    stock_dirs = [d for d in os.listdir(fs.features_dir) if os.path.isdir(os.path.join(fs.features_dir, d))]
    
    if not stock_dirs:
        print("❌ 没有找到特征数据!")
        print("   请先运行特征工程模块构建特征")
        return
    
    print(f"📊 发现 {len(stock_dirs)} 只股票的特征:")
    for i, stock in enumerate(stock_dirs, 1):
        print(f"   {i}. {stock}")
    
    # 选择第一只股票进行分析
    if stock_dirs:
        test_stock = stock_dirs[0]
        print(f"\n🧪 测试特征选择: {test_stock}")
        
        # 获取最优特征集
        optimal_features = fs.get_optimal_feature_set(test_stock)
        
        if optimal_features:
            print(f"\n✅ 特征选择完成!")
            print(f"💡 建议使用 {len(optimal_features)} 个最重要的特征进行模型训练")
        else:
            print(f"\n❌ 特征选择失败")

if __name__ == "__main__":
    main()
