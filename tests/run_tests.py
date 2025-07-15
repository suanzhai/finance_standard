#!/usr/bin/env python3
"""
测试运行脚本
"""

import subprocess
import sys
import os
from pathlib import Path


def run_tests():
    """运行所有测试"""
    
    print("🧪 开始运行金融术语加载工具测试套件")
    print("=" * 50)
    
    # 检查是否安装了pytest
    try:
        import pytest
        print("✅ pytest 已安装")
    except ImportError:
        print("❌ pytest 未安装，正在安装...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest>=7.0.0", "pytest-mock>=3.10.0"])
        print("✅ pytest 安装完成")
    
    # 设置测试环境
    os.environ['PYTHONPATH'] = str(Path(__file__).parent / "src")
    
    # 运行测试
    test_commands = [
        # 快速测试（仅模拟，不需要API）
        ["python", "-m", "pytest", "tests/", "-v", "-m", "not slow"],
        
        # 所有测试
        # ["python", "-m", "pytest", "tests/", "-v"],
        
        # 特定测试文件
        # ["python", "-m", "pytest", "tests/test_configuration.py", "-v"],
        # ["python", "-m", "pytest", "tests/test_finance_term_loader.py", "-v"], 
        # ["python", "-m", "pytest", "tests/test_batch_processing.py", "-v"],
    ]
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n🚀 运行测试命令 {i}: {' '.join(cmd)}")
        print("-" * 40)
        
        try:
            result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=False)
            if result.returncode == 0:
                print(f"✅ 测试命令 {i} 执行成功")
            else:
                print(f"❌ 测试命令 {i} 执行失败，返回码: {result.returncode}")
                return False
        except Exception as e:
            print(f"❌ 测试命令 {i} 执行异常: {e}")
            return False
    
    print("\n🎉 所有测试执行完成！")
    return True


def run_specific_test(test_name):
    """运行特定测试"""
    
    test_files = {
        'config': 'tests/test_configuration.py',
        'loader': 'tests/test_finance_term_loader.py', 
        'batch': 'tests/test_batch_processing.py',
    }
    
    if test_name not in test_files:
        print(f"❌ 未知的测试名称: {test_name}")
        print(f"可用的测试: {', '.join(test_files.keys())}")
        return False
    
    test_file = test_files[test_name]
    print(f"🧪 运行特定测试: {test_file}")
    
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", test_file, "-v"],
            cwd=Path(__file__).parent
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 测试执行异常: {e}")
        return False


def install_dependencies():
    """安装测试依赖"""
    print("📦 安装测试依赖...")
    
    dependencies = [
        "pytest>=7.0.0",
        "pytest-mock>=3.10.0",
        "pandas>=2.0.0",
        "python-dotenv>=1.0.0"
    ]
    
    for dep in dependencies:
        print(f"安装 {dep}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
            print(f"✅ {dep} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ {dep} 安装失败: {e}")
            return False
    
    print("✅ 所有测试依赖安装完成")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="运行金融术语加载工具测试")
    parser.add_argument(
        '--test', 
        choices=['config', 'loader', 'batch'],
        help="运行特定测试"
    )
    parser.add_argument(
        '--install', 
        action='store_true',
        help="安装测试依赖"
    )
    
    args = parser.parse_args()
    
    if args.install:
        success = install_dependencies()
        sys.exit(0 if success else 1)
    
    if args.test:
        success = run_specific_test(args.test)
        sys.exit(0 if success else 1)
    
    # 默认运行所有测试
    success = run_tests()
    sys.exit(0 if success else 1) 