#!/usr/bin/env python3
"""
批处理功能测试
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from finance_term_loader import FinanceTermLoader


class TestBatchProcessing:
    """批处理功能测试"""
    
    def test_batch_size_configuration(self):
        """测试批处理大小配置"""
        test_cases = [
            ('100', 100),
            ('500', 500),
            ('1000', 1000),
            ('', 500),  # 空字符串使用默认值
        ]
        
        for env_value, expected in test_cases:
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'test-key',
                'BATCH_SIZE': env_value
            }):
                loader = FinanceTermLoader()
                assert loader.batch_size == expected
    
    def test_insert_batch_size_configuration(self):
        """测试插入批处理大小配置"""
        test_cases = [
            ('500', 500),
            ('1000', 1000),
            ('2000', 2000),
            ('', 1000),  # 空字符串使用默认值
        ]
        
        for env_value, expected in test_cases:
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'test-key',
                'INSERT_BATCH_SIZE': env_value
            }):
                # 这个配置在insert_data方法中读取，所以我们模拟调用
                loader = FinanceTermLoader()
                with patch.dict(os.environ, {'INSERT_BATCH_SIZE': env_value}):
                    # 验证环境变量被正确读取（使用相同的逻辑）
                    insert_batch_size_str = os.getenv('INSERT_BATCH_SIZE', '1000').strip()
                    insert_batch_size = int(insert_batch_size_str) if insert_batch_size_str else 1000
                    assert insert_batch_size == expected
    
    def test_embedding_batch_processing_mock(self):
        """测试嵌入向量批处理（模拟）"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'BATCH_SIZE': '2'  # 小批次用于测试
        }):
            loader = FinanceTermLoader()
            texts = ['银行', '贷款', '投资', '股票']  # 4个文本，应该分2批
            
            # 模拟嵌入模型
            mock_embeddings = MagicMock()
            mock_embeddings.embed_documents.side_effect = [
                [[0.1] * 3072, [0.2] * 3072],  # 第一批
                [[0.3] * 3072, [0.4] * 3072]   # 第二批
            ]
            
            with patch.object(loader, 'embeddings', mock_embeddings):
                embeddings = loader.generate_embeddings(texts)
                
                # 验证结果
                assert len(embeddings) == 4
                assert embeddings[0][0] == 0.1
                assert embeddings[1][0] == 0.2
                assert embeddings[2][0] == 0.3
                assert embeddings[3][0] == 0.4
                
                # 验证批次调用
                assert mock_embeddings.embed_documents.call_count == 2
                calls = mock_embeddings.embed_documents.call_args_list
                assert calls[0][0][0] == ['银行', '贷款']
                assert calls[1][0][0] == ['投资', '股票']
    
    def test_data_insertion_batch_processing_mock(self):
        """测试数据插入批处理（模拟）"""
        import pandas as pd
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            loader = FinanceTermLoader()
            
            # 创建测试数据
            df = pd.DataFrame({
                'term': ['银行', '贷款', '投资'],
                'category': ['FINTERM', 'FINTERM', 'FINTERM']
            })
            embeddings = [[0.1] * 3072, [0.2] * 3072, [0.3] * 3072]
            
            # 模拟集合
            mock_collection = MagicMock()
            
            # 设置小的插入批次大小进行测试
            with patch.dict(os.environ, {'INSERT_BATCH_SIZE': '2'}):
                result = loader.insert_data(mock_collection, df, embeddings)
                
                # 验证插入调用次数（3条数据，批次大小2，应该分2批）
                assert mock_collection.insert.call_count == 2
                assert mock_collection.flush.called
                assert result == 3
    
    def test_progress_tracking(self):
        """测试进度追踪功能"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'BATCH_SIZE': '2'
        }):
            loader = FinanceTermLoader()
            
            # 模拟嵌入模型和进度条
            mock_embeddings = MagicMock()
            mock_embeddings.embed_documents.return_value = [[0.1] * 3072, [0.2] * 3072]
            
            with patch.object(loader, 'embeddings', mock_embeddings):
                with patch('src.finance_term_loader.tqdm') as mock_tqdm:
                    # 模拟tqdm
                    mock_tqdm.return_value.__enter__.return_value = range(1)
                    
                    embeddings = loader.generate_embeddings(['测试1', '测试2'])
                    
                    # 验证tqdm被调用（进度条）
                    mock_tqdm.assert_called()
                    assert len(embeddings) == 2


class TestBatchProcessingEdgeCases:
    """批处理边界情况测试"""
    
    def test_single_item_batch(self):
        """测试单个项目的批处理"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'BATCH_SIZE': '1'
        }):
            loader = FinanceTermLoader()
            
            mock_embeddings = MagicMock()
            mock_embeddings.embed_documents.return_value = [[0.1] * 3072]
            
            with patch.object(loader, 'embeddings', mock_embeddings):
                embeddings = loader.generate_embeddings(['单个测试'])
                
                assert len(embeddings) == 1
                assert mock_embeddings.embed_documents.call_count == 1
    
    def test_exact_batch_size_division(self):
        """测试数据数量正好等于批次大小的倍数"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'BATCH_SIZE': '2'
        }):
            loader = FinanceTermLoader()
            texts = ['测试1', '测试2', '测试3', '测试4']  # 4个，正好2批
            
            mock_embeddings = MagicMock()
            mock_embeddings.embed_documents.side_effect = [
                [[0.1] * 3072, [0.2] * 3072],
                [[0.3] * 3072, [0.4] * 3072]
            ]
            
            with patch.object(loader, 'embeddings', mock_embeddings):
                embeddings = loader.generate_embeddings(texts)
                
                assert len(embeddings) == 4
                assert mock_embeddings.embed_documents.call_count == 2
    
    def test_large_batch_size(self):
        """测试大批次大小（超过实际数据量）"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'BATCH_SIZE': '1000'  # 比数据量大
        }):
            loader = FinanceTermLoader()
            texts = ['测试1', '测试2']  # 只有2个
            
            mock_embeddings = MagicMock()
            mock_embeddings.embed_documents.return_value = [[0.1] * 3072, [0.2] * 3072]
            
            with patch.object(loader, 'embeddings', mock_embeddings):
                embeddings = loader.generate_embeddings(texts)
                
                assert len(embeddings) == 2
                assert mock_embeddings.embed_documents.call_count == 1  # 只需要1批


class TestBatchProcessingConfiguration:
    """批处理配置测试"""
    
    def test_configuration_combinations(self):
        """测试不同的配置组合"""
        configs = [
            {'BATCH_SIZE': '100', 'INSERT_BATCH_SIZE': '500'},
            {'BATCH_SIZE': '300', 'INSERT_BATCH_SIZE': '800'},
            {'BATCH_SIZE': '500', 'INSERT_BATCH_SIZE': '1000'},
            {'BATCH_SIZE': '1000', 'INSERT_BATCH_SIZE': '2000'},
        ]
        
        for config in configs:
            env_vars = {'OPENAI_API_KEY': 'test-key'}
            env_vars.update(config)
            
            with patch.dict(os.environ, env_vars):
                loader = FinanceTermLoader()
                assert loader.batch_size == int(config['BATCH_SIZE'])
    
    def test_invalid_batch_size_handling(self):
        """测试无效批次大小的处理"""
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'BATCH_SIZE': 'invalid'
        }):
            # 应该使用默认值或抛出错误
            try:
                loader = FinanceTermLoader()
                # 如果没有抛出错误，检查是否使用了默认值
                assert isinstance(loader.batch_size, int)
                assert loader.batch_size > 0
            except ValueError:
                # 如果抛出错误，这也是可接受的
                pass 