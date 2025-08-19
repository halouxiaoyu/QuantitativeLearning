#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试运行脚本
运行所有测试并生成报告
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_tests():
    """运行所有测试"""
    print("🚀 开始运行量化交易系统测试")
    print("=" * 60)
    
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # 创建测试报告目录
    test_reports_dir = os.path.join('tests', 'reports')
    os.makedirs(test_reports_dir, exist_ok=True)
    
    # 生成测试报告文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(test_reports_dir, f'test_report_{timestamp}.html')
    coverage_file = os.path.join(test_reports_dir, f'coverage_report_{timestamp}.html')
    
    print(f"📊 测试报告将保存到: {report_file}")
    print(f"📈 覆盖率报告将保存到: {coverage_file}")
    
    # 检查pytest是否安装
    try:
        import pytest
        print("✅ pytest已安装")
    except ImportError:
        print("❌ pytest未安装，正在安装...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-html", "pytest-cov"])
    
    # 运行单元测试
    print(f"\n🧪 运行单元测试...")
    unit_test_cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/",
        "-v",
        "--tb=short"
    ]
    
    start_time = time.time()
    unit_result = subprocess.run(unit_test_cmd, capture_output=True, text=True)
    unit_time = time.time() - start_time
    
    print(f"   单元测试完成，耗时: {unit_time:.2f}秒")
    print(f"   退出码: {unit_result.returncode}")
    
    # 单独运行HTML报告生成
    if unit_result.returncode == 0:
        html_cmd = [
            sys.executable, "-m", "pytest",
            "tests/unit/",
            f"--html={report_file}",
            "--self-contained-html"
        ]
        subprocess.run(html_cmd, capture_output=True, text=True)
    
    # 运行集成测试
    print(f"\n🔗 运行集成测试...")
    integration_test_cmd = [
        sys.executable, "-m", "pytest",
        "tests/integration/",
        "-v",
        "--tb=short"
    ]
    
    start_time = time.time()
    integration_result = subprocess.run(integration_test_cmd, capture_output=True, text=True)
    integration_time = time.time() - start_time
    
    print(f"   集成测试完成，耗时: {integration_time:.2f}秒")
    print(f"   退出码: {integration_result.returncode}")
    
    # 运行覆盖率测试
    print(f"\n📈 运行覆盖率测试...")
    coverage_cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "--cov=modules",
        f"--cov-report=html:{os.path.dirname(coverage_file)}",
        "--cov-report=term-missing",
        "-v"
    ]
    
    start_time = time.time()
    coverage_result = subprocess.run(coverage_cmd, capture_output=True, text=True)
    coverage_time = time.time() - start_time
    
    print(f"   覆盖率测试完成，耗时: {coverage_time:.2f}秒")
    print(f"   退出码: {coverage_result.returncode}")
    
    # 显示测试结果摘要
    print(f"\n📋 测试结果摘要")
    print("=" * 60)
    
    total_time = unit_time + integration_time + coverage_time
    print(f"总测试时间: {total_time:.2f}秒")
    
    # 检查测试结果
    all_passed = (unit_result.returncode == 0 and 
                  integration_result.returncode == 0 and 
                  coverage_result.returncode == 0)
    
    if all_passed:
        print("🎉 所有测试通过!")
        print("✅ 单元测试: 通过")
        print("✅ 集成测试: 通过")
        print("✅ 覆盖率测试: 通过")
    else:
        print("⚠️ 部分测试失败:")
        if unit_result.returncode != 0:
            print("❌ 单元测试: 失败")
        if integration_result.returncode != 0:
            print("❌ 集成测试: 失败")
        if coverage_result.returncode != 0:
            print("❌ 覆盖率测试: 失败")
    
    # 显示覆盖率信息
    if coverage_result.returncode == 0:
        print(f"\n📊 覆盖率报告已生成: {coverage_file}")
        # 尝试从输出中提取覆盖率信息
        for line in coverage_result.stdout.split('\n'):
            if 'TOTAL' in line and '%' in line:
                print(f"总体覆盖率: {line.strip()}")
                break
    
    # 显示测试报告位置
    if unit_result.returncode == 0:
        print(f"📄 测试报告已生成: {report_file}")
    
    return all_passed

def run_specific_test(test_path):
    """运行特定测试"""
    print(f"🎯 运行特定测试: {test_path}")
    
    cmd = [sys.executable, "-m", "pytest", test_path, "-v"]
    result = subprocess.run(cmd)
    
    return result.returncode == 0

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="运行量化交易系统测试")
    parser.add_argument("--test", help="运行特定测试文件")
    parser.add_argument("--unit", action="store_true", help="只运行单元测试")
    parser.add_argument("--integration", action="store_true", help="只运行集成测试")
    
    args = parser.parse_args()
    
    if args.test:
        # 运行特定测试
        success = run_specific_test(args.test)
        sys.exit(0 if success else 1)
    elif args.unit:
        # 只运行单元测试
        print("🧪 只运行单元测试...")
        cmd = [sys.executable, "-m", "pytest", "tests/unit/", "-v"]
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    elif args.integration:
        # 只运行集成测试
        print("🔗 只运行集成测试...")
        cmd = [sys.executable, "-m", "pytest", "tests/integration/", "-v"]
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    else:
        # 运行所有测试
        success = run_tests()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
