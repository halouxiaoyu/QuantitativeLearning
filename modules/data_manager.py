#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据获取与清洗模块
支持多股票数据下载、清洗和本地存储
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 尝试导入数据源
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

try:
    import baostock as bs
    BAOSTOCK_AVAILABLE = True
except ImportError:
    BAOSTOCK_AVAILABLE = False

class DataManager:
    """数据管理器"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.ensure_directories()
        
        # 预定义的股票池
        self.stock_pools = {
            'bank': {
                'name': '银行股',
                'description': '主要银行股票，包括国有大行和股份制银行',
                'stocks': ['sh.600000', 'sh.600036', 'sh.601398', 'sh.601939', 'sz.000001', 'sh.601988', 'sh.601288']
            },
            'tech': {
                'name': '科技股',
                'description': '科技类股票，包括互联网、软件、芯片等',
                'stocks': ['sz.000002', 'sz.000858', 'sh.600519', 'sz.000568', 'sh.600887', 'sz.000725', 'sh.600584']
            },
            'energy': {
                'name': '能源股',
                'description': '能源类股票，包括石油、电力、煤炭等',
                'stocks': ['sh.600028', 'sh.601857', 'sh.600309', 'sh.600104', 'sh.600887', 'sh.600900', 'sh.601088']
            },
            'consumer': {
                'name': '消费股',
                'description': '消费类股票，包括食品饮料、家电、零售等',
                'stocks': ['sh.600519', 'sz.000858', 'sh.600887', 'sh.000858', 'sz.000568', 'sh.600690', 'sz.000895']
            },
            'pharma': {
                'name': '医药股',
                'description': '医药类股票，包括制药、医疗器械等',
                'stocks': ['sh.600276', 'sh.600867', 'sz.000001', 'sh.600196', 'sz.000963', 'sh.600535', 'sz.000661']
            },
            'all': {
                'name': '全市场',
                'description': '包含所有主要股票的综合池',
                'stocks': ['sh.600000', 'sh.600036', 'sh.601398', 'sh.601939', 'sz.000001', 
                          'sz.000002', 'sz.000858', 'sh.600519', 'sh.000858', 'sz.000568',
                          'sh.600028', 'sh.601857', 'sh.600309', 'sh.600104', 'sh.600887']
            }
        }
        
    def ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            self.data_dir,
            os.path.join(self.data_dir, 'raw'),
            os.path.join(self.data_dir, 'cleaned'),
            os.path.join(self.data_dir, 'metadata')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def show_stock_pools(self):
        """显示可用的股票池"""
        print("📊 可用的股票池:")
        print("=" * 60)
        
        for pool_id, pool_info in self.stock_pools.items():
            print(f"🔸 {pool_id.upper()}: {pool_info['name']}")
            print(f"   描述: {pool_info['description']}")
            print(f"   股票数量: {len(pool_info['stocks'])} 只")
            print(f"   股票列表: {', '.join(pool_info['stocks'])}")
            print("-" * 40)
    
    def get_stock_pool(self, pool_id):
        """获取指定股票池的股票列表"""
        if pool_id in self.stock_pools:
            return self.stock_pools[pool_id]['stocks']
        else:
            print(f"❌ 股票池 '{pool_id}' 不存在")
            return []
    
    def download_stock_data(self, stock_code, start_date='20200101', end_date=None, data_source='auto'):
        """下载单个股票数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            data_source: 数据源选择 ('baostock', 'akshare', 'auto')
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        # 确保结束日期不超过昨天（akshare可能没有今天的数据）
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        if end_date > yesterday:
            end_date = yesterday
            print(f"   ⚠️  结束日期调整为昨天: {end_date}")
        
        print(f"📥 下载股票数据: {stock_code} (数据源: {data_source})")
        
        # 根据选择尝试不同的数据源
        if data_source == 'baostock' or data_source == 'auto':
            df = self._download_with_baostock(stock_code, start_date, end_date)
            if df is not None:
                return df
        
        if data_source == 'akshare' or (data_source == 'auto' and AKSHARE_AVAILABLE):
            # 传递调整后的日期
            df = self._download_with_akshare(stock_code, start_date, end_date)
            if df is not None:
                return df
        
        print(f"❌ 所有数据源都失败: {stock_code}")
        return None
    
    def _download_with_baostock(self, stock_code, start_date, end_date):
        """使用baostock下载数据"""
        if not BAOSTOCK_AVAILABLE:
            print("   ❌ baostock不可用")
            return None
        
        try:
            bs.login()
            start_date_bs = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
            end_date_bs = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
            
            # 确保股票代码格式正确 (sh.600000 或 sz.000001)
            bs_stock_code = stock_code
            if '.' not in stock_code:
                if stock_code.startswith('sh'):
                    bs_stock_code = f"sh.{stock_code[2:]}"
                elif stock_code.startswith('sz'):
                    bs_stock_code = f"sz.{stock_code[2:]}"
            
            rs = bs.query_history_k_data_plus(bs_stock_code,
                                            "date,open,high,low,close,volume",
                                            start_date=start_date_bs,
                                            end_date=end_date_bs,
                                            frequency="d",
                                            adjustflag="3")
            
            if rs.error_code == '0':
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                
                if data_list:
                    df = pd.DataFrame(data_list, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
                    
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    df = df.dropna()
                    
                    print(f"   ✅ baostock下载成功: {len(df)} 条记录")
                    bs.logout()
                    return df
            
            bs.logout()
            
        except Exception as e:
            print(f"   ❌ baostock错误: {e}")
        
        return None
    
    def _download_with_akshare(self, stock_code, start_date, end_date):
        """使用akshare下载数据"""
        if not AKSHARE_AVAILABLE:
            print("   ❌ akshare不可用")
            return None
        
        try:
            # 转换股票代码格式 (akshare使用不带前缀的格式)
            ak_stock_code = stock_code
            if stock_code.startswith('sh.'):
                ak_stock_code = stock_code.replace('sh.', '')
            elif stock_code.startswith('sz.'):
                ak_stock_code = stock_code.replace('sz.', '')
            
            print(f"   🔍 转换后的股票代码: {ak_stock_code}")
            
            # 确保结束日期不超过昨天（部分接口无当日数据）
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
            if end_date > yesterday:
                end_date = yesterday
                print(f"   ⚠️  akshare结束日期调整为昨天: {end_date}")

            # 转换日期格式为YYYYMMDD（akshare要求）
            start_date_ak = start_date.replace('-', '')
            end_date_ak = end_date.replace('-', '')

            print(f"   🔍 实际使用的日期: {start_date_ak} 到 {end_date_ak}")
            print(f"   🔍 尝试akshare下载: stock_zh_a_hist(qfq) {ak_stock_code}, {start_date_ak} 到 {end_date_ak}")

            # 首选：前复权，YYYYMMDD格式
            df = ak.stock_zh_a_hist(
                symbol=ak_stock_code,
                start_date=start_date_ak,
                end_date=end_date_ak,
                adjust="qfq"
            )

            # 回退1：无复权
            if df is None or df.empty:
                print("   ⚠️ qfq返回空，尝试不复权...")
                try:
                    df = ak.stock_zh_a_hist(
                        symbol=ak_stock_code,
                        start_date=start_date_ak,
                        end_date=end_date_ak
                    )
                except Exception as e2:
                    print(f"   ❌ 无复权失败: {e2}")

            # 回退2：使用daily接口
            if df is None or df.empty:
                print("   ⚠️ 再尝试 stock_zh_a_daily...")
                try:
                    df_daily = ak.stock_zh_a_daily(symbol=ak_stock_code, adjust="qfq")
                    if df_daily is not None and not df_daily.empty:
                        # 统一列名
                        if '日期' in df_daily.columns:
                            df_daily = df_daily.rename(columns={'日期': 'date'})
                        elif 'trade_date' in df_daily.columns:
                            df_daily = df_daily.rename(columns={'trade_date': 'date'})
                        elif '时间' in df_daily.columns:
                            df_daily = df_daily.rename(columns={'时间': 'date'})

                        # 其余列中英互转
                        rename_map = {
                            '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume',
                            'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'
                        }
                        df_daily = df_daily.rename(columns={c: rename_map[c] for c in df_daily.columns if c in rename_map})

                        # 过滤日期范围
                        if 'date' in df_daily.columns:
                            df_daily['date'] = pd.to_datetime(df_daily['date'])
                            start_dt = pd.to_datetime(start_date_ak)
                            end_dt = pd.to_datetime(end_date_ak)
                            df_daily = df_daily[(df_daily['date'] >= start_dt) & (df_daily['date'] <= end_dt)]
                            # 仅保留必要列
                            keep_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                            df = df_daily[[c for c in keep_cols if c in df_daily.columns]].copy()
                        else:
                            df = pd.DataFrame()
                except Exception as e3:
                    print(f"   ❌ stock_zh_a_daily失败: {e3}")
            
            if df is not None and not df.empty:
                # 重命名列以保持一致性
                df = df.rename(columns={
                    '日期': 'date',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume'
                })
                
                # 设置索引
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                
                # 确保数据类型正确
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna()
                
                print(f"   ✅ akshare下载成功: {len(df)} 条记录")
                return df
            else:
                print(f"   ❌ akshare返回空数据，可能是网络问题或接口变化")
                
        except Exception as e:
            print(f"   ❌ akshare错误: {e}")
            print(f"   💡 建议使用baostock作为主要数据源")
        
        return None
    
    def clean_data(self, df, stock_code):
        """清洗数据"""
        if df is None or df.empty:
            return None
        
        print(f"🧹 清洗数据: {stock_code}")
        
        # 移除重复和缺失值
        initial_count = len(df)
        df = df.drop_duplicates().dropna()
        
        # 数据质量检查
        price_cols = ['open', 'close', 'high', 'low']
        df = df[df[price_cols].gt(0).all(axis=1)]
        df = df[df['volume'] > 0]
        
        # 排序和添加元数据
        df = df.sort_index()
        df.attrs['stock_code'] = stock_code
        df.attrs['data_points'] = len(df)
        
        # 检查索引是否为日期类型
        try:
            if hasattr(df.index, 'strftime'):
                df.attrs['date_range'] = f"{df.index.min().strftime('%Y-%m-%d')} 到 {df.index.max().strftime('%Y-%m-%d')}"
            else:
                df.attrs['date_range'] = f"索引 {df.index.min()} 到 {df.index.max()}"
        except Exception:
            df.attrs['date_range'] = "日期范围未知"
        
        print(f"   ✅ 清洗完成: {len(df)} 条有效记录")
        return df
    
    def save_data(self, df, stock_code, data_type='cleaned'):
        """保存数据到本地文件"""
        if df is None or df.empty:
            return False
        
        try:
            save_dir = os.path.join(self.data_dir, data_type)
            stock_dir = os.path.join(save_dir, stock_code)
            os.makedirs(stock_dir, exist_ok=True)
            
            try:
                if hasattr(df.index, 'strftime'):
                    start_date = df.index.min().strftime('%Y%m%d')
                    end_date = df.index.max().strftime('%Y%m%d')
                else:
                    start_date = str(df.index.min())
                    end_date = str(df.index.max())
                filename = f"{stock_code}_{start_date}_{end_date}.csv"
            except Exception:
                filename = f"{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join(stock_dir, filename)
            
            df.to_csv(filepath, encoding='utf-8')
            print(f"   💾 数据已保存: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ 保存数据失败: {e}")
            return False
    
    def batch_download(self, stock_list=None, pool_name='all', start_date='20200101', end_date=None, data_source='auto'):
        """批量下载股票数据
        
        Args:
            stock_list: 股票列表
            pool_name: 股票池名称
            start_date: 开始日期
            end_date: 结束日期
            data_source: 数据源选择 ('baostock', 'akshare', 'auto')
        """
        if stock_list is None:
            stock_list = self.get_stock_list(pool_name)
        
        print(f"🚀 开始批量下载 {len(stock_list)} 只股票 (数据源: {data_source})")
        
        results = {}
        success_count = 0
        
        for stock_code in stock_list:
            print(f"\n📊 处理股票: {stock_code}")
            
            try:
                # 确保日期参数正确传递
                current_end_date = end_date
                if current_end_date is None:
                    current_end_date = datetime.now().strftime('%Y%m%d')
                
                df_raw = self.download_stock_data(stock_code, start_date, current_end_date, data_source)
                
                if df_raw is not None and not df_raw.empty:
                    self.save_data(df_raw, stock_code, 'raw')
                    df_cleaned = self.clean_data(df_raw, stock_code)
                    
                    if df_cleaned is not None:
                        if self.save_data(df_cleaned, stock_code, 'cleaned'):
                            results[stock_code] = {'status': 'success', 'records': len(df_cleaned)}
                            success_count += 1
                
            except Exception as e:
                print(f"   ❌ 处理失败: {e}")
                results[stock_code] = {'status': 'error'}
        
        print(f"\n🎉 批量下载完成! 成功: {success_count}/{len(stock_list)}")
        return results
    
    def get_stock_list(self, pool_name='all'):
        """获取股票列表"""
        if pool_name in self.stock_pools:
            return self.stock_pools[pool_name]['stocks']
        else:
            print(f"❌ 股票池 '{pool_name}' 不存在")
            return []

def main():
    """主函数"""
    import sys
    
    print("📥 数据获取与清洗模块")
    print("=" * 60)
    
    dm = DataManager()
    
    print("🔧 系统状态:")
    print(f"   数据目录: {os.path.abspath(dm.data_dir)}")
    print(f"   baostock: {'✅ 可用' if BAOSTOCK_AVAILABLE else '❌ 不可用'}")
    print(f"   akshare: {'✅ 可用' if AKSHARE_AVAILABLE else '❌ 不可用'}")
    
    # 检查命令行参数
    pool_name = 'bank'  # 默认银行股
    data_source = 'auto'  # 默认自动选择
    
    if len(sys.argv) > 1:
        pool_name = sys.argv[1]
    if len(sys.argv) > 2:
        data_source = sys.argv[2]
    
    print(f"\n🎯 使用参数:")
    print(f"   股票池: {pool_name}")
    print(f"   数据源: {data_source}")
    
    # 显示可用的股票池
    print(f"\n📊 可用的股票池:")
    dm.show_stock_pools()
    
    # 验证股票池是否存在
    if pool_name not in dm.stock_pools:
        print(f"\n❌ 股票池 '{pool_name}' 不存在，使用默认银行股")
        pool_name = 'bank'
    
    # 验证数据源是否可用
    if data_source == 'akshare' and not AKSHARE_AVAILABLE:
        print(f"\n❌ akshare不可用，自动切换到baostock")
        data_source = 'baostock'
    elif data_source == 'baostock' and not BAOSTOCK_AVAILABLE:
        print(f"\n❌ baostock不可用，自动切换到akshare")
        data_source = 'akshare'
    
    # 开始下载
    print(f"\n🚀 开始下载 {dm.stock_pools[pool_name]['name']} (数据源: {data_source})...")
    print("=" * 60)
    
    results = dm.batch_download(
        pool_name=pool_name, 
        start_date='20240101', 
        data_source=data_source
    )
    
    # 显示结果
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    total_count = len(results)
    
    print(f"\n📋 下载结果汇总:")
    print(f"   {dm.stock_pools[pool_name]['name']}: {success_count}/{total_count} 成功")
    
    if success_count == total_count:
        print("🎉 所有股票下载成功！")
    else:
        print(f"⚠️  {total_count - success_count} 只股票下载失败")
        for stock, result in results.items():
            if result['status'] != 'success':
                print(f"   ❌ {stock}: {result.get('error', '未知错误')}")

if __name__ == "__main__":
    main()
