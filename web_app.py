#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI量化交易学习平台 - Web界面
专业的分层架构，适合初学者学习
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 导入功能模块
from modules.data_manager import DataManager
from modules.feature_engineer import FeatureEngineer
from modules.model_trainer import ModelTrainer
from modules.backtest_analyzer import BacktestAnalyzer
from modules.historical_predictor import HistoricalPredictor  # 历史预测验证器
from modules.future_predictor import FuturePredictor  # 未来预测器

app = Flask(__name__)

# 全局变量存储结果
app.last_data_results = None
app.last_feature_results = None
app.last_training_results = None
app.last_backtest_results = None
app.last_historical_validation_results = None  # 历史验证结果
app.last_future_prediction_results = None      # 未来预测结果

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

# ==================== 数据管理模块 ====================

@app.route('/api/data/status')
def data_status():
    """获取数据状态"""
    try:
        dm = DataManager()
        
        # 检查数据目录
        data_info = {
            'raw_data': {},
            'cleaned_data': {},
            'stock_pools': dm.stock_pools
        }
        
        # 检查原始数据
        raw_dir = os.path.join(dm.data_dir, 'raw')
        if os.path.exists(raw_dir):
            stock_dirs = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
            for stock_dir in stock_dirs:
                stock_path = os.path.join(raw_dir, stock_dir)
                files = [f for f in os.listdir(stock_path) if f.endswith('.csv')]
                if files:
                    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(stock_path, x)))
                    file_path = os.path.join(stock_path, latest_file)
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    data_info['raw_data'][stock_dir] = {
                        'records': len(df),
                        'date_range': f"{df.index.min().strftime('%Y-%m-%d')} 到 {df.index.max().strftime('%Y-%m-%d')}",
                        'latest_file': latest_file
                    }
        
        # 检查清洗后的数据
        cleaned_dir = os.path.join(dm.data_dir, 'cleaned')
        if os.path.exists(cleaned_dir):
            stock_dirs = [d for d in os.listdir(cleaned_dir) if os.path.isdir(os.path.join(cleaned_dir, d))]
            for stock_dir in stock_dirs:
                stock_path = os.path.join(cleaned_dir, stock_dir)
                files = [f for f in os.listdir(stock_path) if f.endswith('.csv')]
                if files:
                    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(stock_path, x)))
                    file_path = os.path.join(stock_path, latest_file)
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    data_info['cleaned_data'][stock_dir] = {
                        'records': len(df),
                        'date_range': f"{df.index.min().strftime('%Y-%m-%d')} 到 {df.index.max().strftime('%Y-%m-%d')}",
                        'latest_file': latest_file
                    }
        
        return jsonify({'success': True, 'data': data_info})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/data/download', methods=['POST'])
def download_data():
    """下载股票数据"""
    try:
        data = request.get_json()
        pool_name = data.get('pool_name', 'bank')
        data_source = data.get('data_source', 'auto')
        start_date = data.get('start_date', '20210101')
        end_date = data.get('end_date', '20251231')
        
        print(f"📥 开始下载数据: 股票池={pool_name}, 数据源={data_source}, 日期范围={start_date}-{end_date}")
        
        dm = DataManager()
        results = dm.batch_download(pool_name=pool_name, start_date=start_date, 
                                  end_date=end_date, data_source=data_source)
        
        app.last_data_results = results
        
        return jsonify({
            'success': True,
            'message': f'数据下载完成 (数据源: {data_source}, 时间: {start_date}-{end_date})，成功: {sum(1 for r in results.values() if r["status"] == "success")}/{len(results)}',
            'results': results
        })
        
    except Exception as e:
        print(f"❌ 数据下载失败: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/data/download_single', methods=['POST'])
def download_single_stock():
    """下载单个股票数据"""
    try:
        data = request.get_json()
        stock_code = data.get('stock_code', '').strip()
        data_source = data.get('data_source', 'auto')
        start_date = data.get('start_date', '20210101')
        end_date = data.get('end_date', '20251231')
        
        if not stock_code:
            return jsonify({'success': False, 'error': '请输入股票代码'})
        
        print(f"📥 开始下载单个股票: {stock_code}, 数据源={data_source}, 日期范围={start_date}-{end_date}")
        
        dm = DataManager()
        
        # 验证股票代码
        validation = dm.validate_stock_code(stock_code)
        if not validation['valid']:
            return jsonify({'success': False, 'error': f'股票代码验证失败：{validation["error"]}\n💡 请检查代码是否正确，或尝试其他格式'})
        
        # 下载单个股票数据
        result = dm.download_single_stock(stock_code, start_date, end_date, data_source)
        
        if result['status'] == 'success':
            normalized_code = validation.get('normalized_code', stock_code)
            app.last_data_results = {normalized_code: result}
            return jsonify({
                'success': True,
                'message': f'✅ 股票 {normalized_code} 数据下载完成！\n📊 数据范围：{start_date} 到 {end_date}\n💾 已保存到本地',
                'result': result
            })
        else:
            return jsonify({
                'success': False, 'error': f'❌ 股票 {stock_code} 下载失败\n🔍 错误详情：{result.get("error", "未知错误")}\n💡 请检查网络连接或稍后重试'
            })
        
    except Exception as e:
        print(f"❌ 单个股票下载失败: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/data/validate_stock', methods=['POST'])
def validate_stock_code():
    """验证股票代码格式"""
    try:
        data = request.get_json()
        stock_code = data.get('stock_code', '').strip()
        
        if not stock_code:
            return jsonify({'success': False, 'error': '请输入股票代码'})
        
        dm = DataManager()
        validation = dm.validate_stock_code(stock_code)
        
        # 优化验证结果提示
        if validation['valid']:
            normalized_code = validation.get('normalized_code', stock_code)
            exchange_info = {
                'sh.': '上海证券交易所',
                'sz.': '深圳证券交易所',
                'bj.': '北京证券交易所'
            }
            exchange = next((info for prefix, info in exchange_info.items() if normalized_code.startswith(prefix)), '未知交易所')
            board = validation.get('board', '未知板块')
            
            return jsonify({
                'success': True,
                'validation': validation,
                'user_friendly_message': f'✅ 股票代码有效！\n📊 标准格式：{normalized_code}\n🏢 交易所：{exchange}\n📈 板块：{board}'
            })
        else:
            return jsonify({
                'success': True,
                'validation': validation,
                'user_friendly_message': f'❌ 股票代码无效：{validation.get("error", "未知错误")}'
            })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/data/clear', methods=['POST'])
def clear_all_data():
    """清空所有数据"""
    try:
        print("🗑️ 开始清空所有数据...")
        
        # 需要确认
        data = request.get_json()
        confirm = data.get('confirm', False)
        
        if not confirm:
            return jsonify({'success': False, 'error': '需要确认才能清空数据'})
        
        # 清空各个目录
        directories_to_clear = [
            'data/raw',
            'data/cleaned', 
            'features',
            'models',
            'results'
        ]
        
        cleared_count = 0
        total_size = 0
        
        for directory in directories_to_clear:
            if os.path.exists(directory):
                try:
                    # 计算目录大小
                    dir_size = 0
                    for root, dirs, files in os.walk(directory):
                        for file in files:
                            file_path = os.path.join(root, file)
                            if os.path.exists(file_path):
                                dir_size += os.path.getsize(file_path)
                    
                    # 删除目录内容
                    import shutil
                    shutil.rmtree(directory)
                    os.makedirs(directory, exist_ok=True)
                    
                    cleared_count += 1
                    total_size += dir_size
                    
                    print(f"   ✅ 清空目录: {directory} (大小: {dir_size / (1024*1024):.2f} MB)")
                    
                except Exception as e:
                    print(f"   ❌ 清空目录失败: {directory} - {e}")
        
        # 重置工作流程状态
        app.last_data_results = None
        app.last_feature_results = None
        app.last_training_results = None
        app.last_backtest_results = None
        app.last_historical_validation_results = None
        app.last_future_prediction_results = None
        
        print(f"🗑️ 数据清空完成: {cleared_count} 个目录, 总大小: {total_size / (1024*1024):.2f} MB")
        
        return jsonify({
            'success': True,
            'message': f'数据清空完成！清空了 {cleared_count} 个目录',
            'cleared_directories': cleared_count,
            'total_size_mb': round(total_size / (1024*1024), 2)
        })
        
    except Exception as e:
        print(f"❌ 清空数据失败: {e}")
        return jsonify({'success': False, 'error': str(e)})

# ==================== 特征工程模块 ====================

@app.route('/api/features/status')
def features_status():
    """获取特征状态"""
    try:
        fe = FeatureEngineer()
        
        features_info = {}
        if os.path.exists(fe.features_dir):
            stock_dirs = [d for d in os.listdir(fe.features_dir) if os.path.isdir(os.path.join(fe.features_dir, d))]
            for stock_dir in stock_dirs:
                stock_path = os.path.join(fe.features_dir, stock_dir)
                files = [f for f in os.listdir(stock_path) if f.endswith('.csv')]
                if files:
                    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(stock_path, x)))
                    file_path = os.path.join(stock_path, latest_file)
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    features_info[stock_dir] = {
                        'features': len(df.columns),
                        'records': len(df),
                        'date_range': f"{df.index.min().strftime('%Y-%m-%d')} 到 {df.index.max().strftime('%Y-%m-%d')}",
                        'latest_file': latest_file
                    }
        
        return jsonify({'success': True, 'data': features_info})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/features/build', methods=['POST'])
def build_features():
    """构建特征"""
    try:
        fe = FeatureEngineer()
        results = fe.batch_build_features()
        
        app.last_feature_results = results
        
        return jsonify({
            'success': True,
            'message': f'特征构建完成，成功: {sum(1 for r in results.values() if r["status"] == "success")}/{len(results)}',
            'results': results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ==================== 模型训练模块 ====================

@app.route('/api/models/status')
def models_status():
    """获取模型状态"""
    try:
        mt = ModelTrainer(
            features_dir='features',
            models_dir='models',
            results_dir='results'
        )
        
        models_info = {}
        if os.path.exists(mt.models_dir):
            stock_dirs = [d for d in os.listdir(mt.models_dir) if os.path.isdir(os.path.join(mt.models_dir, d))]
            print(f"Found stock directories: {stock_dirs}")
            
            for stock_dir in stock_dirs:
                stock_path = os.path.join(mt.models_dir, stock_dir)
                info_files = [f for f in os.listdir(stock_path) if '_info_' in f and f.endswith('.json')]
                print(f"Stock {stock_dir}: found info files: {info_files}")
                
                if info_files:
                    latest_info = max(info_files, key=lambda x: os.path.getctime(os.path.join(stock_path, x)))
                    info_path = os.path.join(stock_path, latest_info)
                    print(f"Stock {stock_dir}: using latest info file: {latest_info}")
                    
                    try:
                        with open(info_path, 'r', encoding='utf-8') as f:
                            model_info = json.load(f)
                        
                        models_info[stock_dir] = {
                            'training_date': model_info.get('training_date', 'Unknown'),
                            'algorithm': model_info.get('algorithm', 'Unknown'),
                            'feature_count': model_info.get('data_info', {}).get('feature_count', 0),
                            'cv_accuracy': f"{model_info.get('cv_scores', {}).get('mean', 0):.3f} ± {model_info.get('cv_scores', {}).get('std', 0):.3f}",
                            'train_accuracy': f"{model_info.get('training_metrics', {}).get('train_accuracy', 0):.3f}",
                            'train_f1': f"{model_info.get('training_metrics', {}).get('train_f1', 0):.3f}",
                            'test_accuracy': f"{model_info.get('training_metrics', {}).get('test_accuracy', 0):.3f}",
                            'test_f1': f"{model_info.get('training_metrics', {}).get('test_f1', 0):.3f}",
                            'training_samples': model_info.get('data_info', {}).get('training_samples', 0),
                            'test_samples': model_info.get('data_info', {}).get('test_samples', 0)
                        }
                        print(f"Stock {stock_dir}: model info loaded successfully")
                    except Exception as e:
                        print(f"Error reading model info for {stock_dir}: {e}")
                        # 如果读取失败，至少显示有模型文件
                        models_info[stock_dir] = {
                            'training_date': 'Available',
                            'algorithm': 'Unknown',
                            'feature_count': 'Unknown',
                            'cv_accuracy': 'N/A',
                            'cv_f1': 'N/A'
                        }
                else:
                    print(f"Stock {stock_dir}: no info files found")
        
        print(f"Final models_info: {models_info}")
        return jsonify({'success': True, 'data': models_info})
        
    except Exception as e:
        print(f"Error in models_status: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/models/algorithms')
def get_available_algorithms():
    """获取可用的算法列表"""
    try:
        mt = ModelTrainer(
            features_dir='features',
            models_dir='models',
            results_dir='results'
        )
        
        algorithms = mt.get_available_algorithms()
        return jsonify({'success': True, 'data': algorithms})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/models/train', methods=['POST'])
def train_models():
    """训练模型"""
    try:
        data = request.get_json()
        algorithm = data.get('algorithm', 'random_forest')
        cv_folds = int(data.get('cv_folds', 5))
        random_seed = int(data.get('random_seed', 42))
        start_date = data.get('start_date', '20230207')
        end_date = data.get('end_date', '20240601')
        
        print(f"训练参数: algorithm={algorithm}, cv_folds={cv_folds}, random_seed={random_seed}")
        print(f"训练日期: {start_date} 到 {end_date}")
        
        mt = ModelTrainer(
            features_dir='features',
            models_dir='models',
            results_dir='results'
        )
        
        results = mt.batch_train_models(
            algorithm=algorithm,
            cv_folds=cv_folds,
            start_date=start_date,
            end_date=end_date,
            random_seed=random_seed
        )
        
        app.last_training_results = results
        
        return jsonify({
            'success': True,
            'message': f'模型训练完成，成功: {sum(1 for r in results.values() if r.get("success", False))}/{len(results)}',
            'results': results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ==================== 回测分析模块 ====================

@app.route('/api/backtest/status')
def backtest_status():
    """获取回测状态"""
    try:
        ba = BacktestAnalyzer()
        
        backtest_info = {}
        backtest_dir = os.path.join(ba.results_dir, 'backtests')
        if os.path.exists(backtest_dir):
            stock_dirs = [d for d in os.listdir(backtest_dir) if os.path.isdir(os.path.join(backtest_dir, d))]
            for stock_dir in stock_dirs:
                stock_path = os.path.join(backtest_dir, stock_dir)
                files = [f for f in os.listdir(stock_path) if f.endswith('_backtest_.json')]
                if files:
                    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(stock_path, x)))
                    backtest_info[stock_dir] = {
                        'latest_backtest': latest_file,
                        'backtest_count': len(files)
                    }
        
        return jsonify({'success': True, 'data': backtest_info})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/backtest/run', methods=['POST'])
def run_backtest():
    """运行回测"""
    try:
        data = request.get_json()
        cash = float(data.get('cash', 100000))
        commission = float(data.get('commission', 0.0008))
        ml_threshold = float(data.get('ml_threshold', 0.51))
        start_date = data.get('start_date', '20240101')
        end_date = data.get('end_date', '20240801')
        
        print(f"回测参数: cash={cash}, commission={commission}, ml_threshold={ml_threshold}")
        print(f"回测日期: {start_date} 到 {end_date}")
        
        ba = BacktestAnalyzer()
        results = ba.batch_backtest(
            cash=cash, 
            commission=commission, 
            ml_threshold=ml_threshold,
            start_date=start_date,
            end_date=end_date
        )
        
        app.last_backtest_results = results
        
        return jsonify({
            'success': True,
            'message': f'回测完成，成功: {sum(1 for r in results.values() if isinstance(r, dict) and "ml_strategy" in r)}/{len(results)}',
            'results': results
        })
        
    except Exception as e:
        print(f"回测错误: {e}")
        return jsonify({'success': False, 'error': str(e)})

# ==================== 预测模块 ====================

@app.route('/api/prediction/status')
def prediction_status():
    """获取预测状态"""
    try:
        # 注意：这里应该使用HistoricalPredictor或FuturePredictor
        # 根据具体需求选择
        print("请使用新的预测系统：")
        print("- HistoricalPredictor: 历史验证")
        print("- FuturePredictor: 未来预测")
        
        prediction_info = {}
        prediction_dir = os.path.join(ps.results_dir, 'predictions')
        if os.path.exists(prediction_dir):
            stock_dirs = [d for d in os.listdir(prediction_dir) if os.path.isdir(os.path.join(prediction_dir, d))]
            for stock_dir in stock_dirs:
                stock_path = os.path.join(prediction_dir, stock_dir)
                files = [f for f in os.listdir(stock_path) if f.endswith('_prediction_.csv')]
                if files:
                    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(stock_path, x)))
                    prediction_info[stock_dir] = {
                        'latest_prediction': latest_file,
                        'prediction_count': len(files)
                    }
        
        return jsonify({'success': True, 'data': prediction_info})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ==================== 历史预测验证模块 ====================

@app.route('/api/historical/status')
def historical_validation_status():
    """获取历史验证状态"""
    try:
        hp = HistoricalPredictor('features', 'models', 'results')
        
        # 检查历史验证结果
        validation_info = {}
        validation_dir = os.path.join('results', 'reports')
        if os.path.exists(validation_dir):
            files = [f for f in os.listdir(validation_dir) if f.startswith('historical_validation_summary_')]
            if files:
                latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(validation_dir, x)))
                validation_info['latest_validation'] = latest_file
                validation_info['validation_count'] = len(files)
        
        return jsonify({'success': True, 'data': validation_info})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/historical/run', methods=['POST'])
def run_historical_validation():
    """运行历史预测验证"""
    try:
        data = request.get_json()
        start_date = data.get('start_date', None)
        end_date = data.get('end_date', None)
        
        hp = HistoricalPredictor('features', 'models', 'results')
        results = hp.batch_validate_historical(start_date=start_date, end_date=end_date)
        
        app.last_historical_validation_results = results
        
        return jsonify({
            'success': True,
            'message': f'历史验证完成，成功: {sum(1 for r in results.values() if r.get("success", False))}/{len(results)}',
            'results': results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ==================== 未来预测模块 ====================

@app.route('/api/future/status')
def future_prediction_status():
    """获取未来预测状态"""
    try:
        fp = FuturePredictor('features', 'models', 'results')
        
        # 检查未来预测结果
        prediction_info = {}
        prediction_dir = os.path.join('results', 'future_predictions')
        if os.path.exists(prediction_dir):
            files = [f for f in os.listdir(prediction_dir) if f.endswith('.csv')]
            if files:
                latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(prediction_dir, x)))
                prediction_info['latest_prediction'] = latest_file
                prediction_info['prediction_count'] = len(files)
        
        return jsonify({'success': True, 'data': prediction_info})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/future/predict', methods=['POST'])
def run_future_prediction():
    """运行未来预测"""
    try:
        data = request.get_json()
        stock_code = data.get('stock_code', None)
        prediction_days = data.get('prediction_days', 5)  # 默认5天
        confidence_threshold = data.get('confidence_threshold', 0.6)
        
        if not stock_code:
            return jsonify({'success': False, 'error': '请指定股票代码'})
        
        # 限制预测天数
        if prediction_days > 5:
            prediction_days = 5
        
        fp = FuturePredictor('features', 'models', 'results')
        result = fp.predict_next_n_days(stock_code, n_days=prediction_days, confidence_threshold=confidence_threshold)
        
        app.last_future_prediction_results = result
        
        if result:
            return jsonify({
                'success': True,
                'message': f'未来预测完成，预测{result["total_predictions"]}天',
                'results': result
            })
        else:
            return jsonify({'success': False, 'error': '预测失败'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/future/batch_predict', methods=['POST'])
def run_batch_future_prediction():
    """批量未来预测"""
    try:
        data = request.get_json()
        stock_list = data.get('stock_list', [])
        prediction_days = data.get('prediction_days', 5)  # 默认5天
        confidence_threshold = data.get('confidence_threshold', 0.6)
        
        if not stock_list:
            return jsonify({'success': False, 'error': '请指定股票列表'})
        
        # 限制预测天数
        if prediction_days > 5:
            prediction_days = 5
        
        fp = FuturePredictor('features', 'models', 'results')
        results = fp.predict_next_n_days(stock_list[0], n_days=prediction_days, confidence_threshold=confidence_threshold)
        
        app.last_future_prediction_results = results
        
        return jsonify({
            'success': True,
            'message': f'批量未来预测完成，预测{prediction_days}天',
            'results': results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ==================== 系统信息 ====================

@app.route('/api/system/info')
def system_info():
    """获取系统信息"""
    try:
        info = {
            'platform_name': 'AI量化交易学习平台',
            'version': '1.0.0',
            'description': '专业的分层架构，适合初学者学习',
            'modules': [
                {
                    'name': '数据获取与清洗',
                    'description': '支持多股票数据下载、清洗和本地存储',
                    'status': 'active'
                },
                {
                    'name': '特征工程',
                    'description': '构建技术指标特征，为机器学习模型提供输入',
                    'status': 'active'
                },
                {
                    'name': '模型训练与保存',
                    'description': '支持多股票模型训练、验证和保存',
                    'status': 'active'
                },
                {
                    'name': '回测与结果分析',
                    'description': '支持多股票回测、结果分析和可视化',
                    'status': 'active'
                },
                {
                    'name': '历史预测验证',
                    'description': '在历史数据上验证模型效果，评估模型表现',
                    'status': 'active'
                },
                {
                    'name': '未来预测',
                    'description': '预测未来1-5天的股票涨跌，生成交易信号',
                    'status': 'active'
                }
            ],
            'directories': {
                'data': os.path.abspath('data'),
                'features': os.path.abspath('features'),
                'models': os.path.abspath('models'),
                'results': os.path.abspath('results')
            }
        }
        
        return jsonify({'success': True, 'data': info})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("🚀 启动AI量化交易学习平台...")
    print("=" * 60)
    print("📚 专业分层架构，适合初学者学习")
    print("=" * 60)
    
    # 确保必要的目录存在
    os.makedirs('data', exist_ok=True)
    os.makedirs('features', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("✅ 系统初始化完成")
    print("🌐 启动Web界面...")
    print("📱 访问地址: http://localhost:5002")
    
    app.run(host='0.0.0.0', port=5002, debug=True)
