#!/usr/bin/env python3
"""
配置管理测试
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from finance_term_loader import FinanceTermLoader


class TestConfiguration:
    """配置管理相关测试"""
    
    def test_environment_variable_loading(self):
        """测试环境变量加载"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-api-key',
            'MILVUS_HOST': 'test-host',
            'MILVUS_PORT': '19530',
            'BATCH_SIZE': '100',
            'TEST_MODE_LIMIT': '50'
        }):
            loader = FinanceTermLoader()
            
            assert loader.openai_api_key == 'test-api-key'
            assert loader.milvus_host == 'test-host'
            assert loader.milvus_port == 19530
            assert loader.batch_size == 100
            assert loader.test_mode_limit == 50
    
    def test_empty_test_mode_limit_handling(self):
        """测试空字符串TEST_MODE_LIMIT的处理"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-api-key',
            'TEST_MODE_LIMIT': ''  # 空字符串，应该转换为0
        }):
            loader = FinanceTermLoader()
            assert loader.test_mode_limit == 0
    
    def test_missing_api_key_raises_error(self):
        """测试缺少API密钥时抛出错误"""
        # 清除API密钥环境变量
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API密钥未提供"):
                FinanceTermLoader()
    
    def test_parameter_override_environment(self):
        """测试参数覆盖环境变量"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'env-api-key',
            'MILVUS_HOST': 'env-host',
            'MILVUS_PORT': '19530'
        }):
            loader = FinanceTermLoader(
                openai_api_key='param-api-key',
                milvus_host='param-host',
                milvus_port=8080
            )
            
            assert loader.openai_api_key == 'param-api-key'
            assert loader.milvus_host == 'param-host'
            assert loader.milvus_port == 8080
    
    def test_default_values(self):
        """测试默认配置值"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            loader = FinanceTermLoader()
            
            assert loader.milvus_host == 'localhost'
            assert loader.milvus_port == 19530
            assert loader.collection_name == 'finance_term'
            assert loader.embedding_model == 'text-embedding-3-large'
            assert loader.embedding_dim == 3072
            assert loader.batch_size == 500
            assert loader.test_mode_limit == 0
    
    def test_config_file_loading(self):
        """测试配置文件加载"""
        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write('OPENAI_API_KEY=file-api-key\n')
            f.write('BATCH_SIZE=200\n')
            f.write('MILVUS_HOST=file-host\n')
            config_file = f.name
        
        try:
            # 使用dotenv加载配置文件
            from dotenv import load_dotenv
            load_dotenv(config_file)
            
            loader = FinanceTermLoader()
            assert loader.openai_api_key == 'file-api-key'
            assert loader.batch_size == 200
            assert loader.milvus_host == 'file-host'
            
        finally:
            # 清理临时文件
            os.unlink(config_file)
    
    def test_milvus_lite_configuration(self):
        """测试Milvus Lite配置"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'MILVUS_USE_LITE': 'true'
        }):
            loader = FinanceTermLoader()
            # 这里我们只测试配置加载，不测试实际连接
            assert loader.openai_api_key == 'test-key'
    
    def test_batch_size_configuration(self):
        """测试批处理大小配置"""
        test_cases = [
            ('100', 100),
            ('500', 500),
            ('1000', 1000),
            ('', 500),  # 空字符串应该使用默认值
        ]
        
        for env_value, expected in test_cases:
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'test-key',
                'BATCH_SIZE': env_value
            }):
                loader = FinanceTermLoader()
                assert loader.batch_size == expected


class TestConfigurationValidation:
    """配置验证测试"""
    
    def test_invalid_port_number(self):
        """测试无效端口号"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'MILVUS_PORT': 'invalid'
        }):
            # 应该在初始化时抛出错误或使用默认值
            loader = FinanceTermLoader()
            # 如果端口无效，应该使用默认值
            assert isinstance(loader.milvus_port, int)
    
    def test_invalid_batch_size(self):
        """测试无效批处理大小"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'BATCH_SIZE': 'invalid'
        }):
            # 应该在初始化时抛出错误或使用默认值
            loader = FinanceTermLoader()
            assert isinstance(loader.batch_size, int)
            assert loader.batch_size > 0 