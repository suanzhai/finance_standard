#!/usr/bin/env python3
"""
FinanceTermLoader 核心功能测试
"""

import os
import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import sys

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from finance_term_loader import FinanceTermLoader


class TestFinanceTermLoader:
    """FinanceTermLoader主要功能测试"""
    
    @pytest.fixture
    def mock_loader(self):
        """创建模拟的FinanceTermLoader实例"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            return FinanceTermLoader()
    
    @pytest.fixture
    def sample_csv_data(self):
        """创建示例CSV数据"""
        return pd.DataFrame({
            'term': ['银行', '贷款', '投资', '股票', '债券'],
            'category': ['FINTERM', 'FINTERM', 'FINTERM', 'FINTERM', 'FINTERM']
        })
    
    def test_csv_reading(self, mock_loader, sample_csv_data):
        """测试CSV文件读取功能"""
        # 创建临时CSV文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            sample_csv_data.to_csv(f.name, index=False, encoding='utf-8')
            csv_path = f.name
        
        try:
            df = mock_loader.read_csv_data(csv_path)
            
            assert len(df) == 5
            assert 'term' in df.columns
            assert 'category' in df.columns
            assert df['term'].iloc[0] == '银行'
            assert all(df['category'] == 'FINTERM')
            
        finally:
            os.unlink(csv_path)
    
    def test_csv_reading_with_test_mode(self, mock_loader, sample_csv_data):
        """测试测试模式下的CSV读取"""
        with patch.dict(os.environ, {'TEST_MODE_LIMIT': '3'}):
            loader = FinanceTermLoader()
            
            # 创建临时CSV文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
                sample_csv_data.to_csv(f.name, index=False, encoding='utf-8')
                csv_path = f.name
            
            try:
                df = loader.read_csv_data(csv_path)
                
                # 应该只读取前3条记录
                assert len(df) == 3
                assert df['term'].iloc[0] == '银行'
                assert df['term'].iloc[2] == '投资'
                
            finally:
                os.unlink(csv_path)
    
    def test_csv_reading_missing_file(self, mock_loader):
        """测试读取不存在的CSV文件"""
        with pytest.raises(Exception):  # 可能是FileNotFoundError或pandas相关错误
            mock_loader.read_csv_data("nonexistent.csv")
    
    def test_embedding_generation_mock(self, mock_loader):
        """测试嵌入向量生成（使用模拟）"""
        texts = ['银行', '贷款', '投资']
        
        # 模拟LangChain嵌入模型
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [
            [0.1] * 3072,  # 模拟3072维向量
            [0.2] * 3072,
            [0.3] * 3072
        ]
        
        with patch.object(mock_loader, 'embeddings', mock_embeddings):
            embeddings = mock_loader.generate_embeddings(texts)
            
            assert len(embeddings) == 3
            assert len(embeddings[0]) == 3072
            assert embeddings[0][0] == 0.1
            assert embeddings[1][0] == 0.2
            assert embeddings[2][0] == 0.3
            
            # 验证调用
            mock_embeddings.embed_documents.assert_called()
    
    @patch('src.finance_term_loader.connections')
    def test_milvus_connection_lite(self, mock_connections, mock_loader):
        """测试Milvus Lite连接"""
        with patch.dict(os.environ, {'MILVUS_USE_LITE': 'true'}):
            mock_loader.connect_milvus()
            
            # 验证连接调用
            mock_connections.connect.assert_called_once()
            call_args = mock_connections.connect.call_args
            assert 'uri' in call_args.kwargs
            assert call_args.kwargs['uri'].endswith('milvus_lite.db')
    
    @patch('src.finance_term_loader.connections')
    def test_milvus_connection_server(self, mock_connections, mock_loader):
        """测试Milvus服务器连接"""
        with patch.dict(os.environ, {'MILVUS_USE_LITE': 'false'}):
            mock_loader.connect_milvus()
            
            # 验证连接调用
            mock_connections.connect.assert_called_once()
            call_args = mock_connections.connect.call_args
            assert call_args.kwargs['host'] == mock_loader.milvus_host
            assert call_args.kwargs['port'] == mock_loader.milvus_port
    
    @patch('src.finance_term_loader.Collection')
    def test_collection_creation(self, mock_collection_class, mock_loader):
        """测试集合创建"""
        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection
        
        # 模拟集合不存在
        with patch('src.finance_term_loader.utility.has_collection', return_value=False):
            collection = mock_loader.create_collection()
            
            assert collection == mock_collection
            mock_collection_class.assert_called_once()
    
    def test_batch_processing_configuration(self):
        """测试批处理配置"""
        test_cases = [
            ('100', '500', 100, 500),
            ('200', '800', 200, 800),
            ('', '1000', 500, 1000),  # 默认BATCH_SIZE
            ('300', '', 300, 1000),   # 默认INSERT_BATCH_SIZE
        ]
        
        for batch_size, insert_batch_size, expected_batch, expected_insert in test_cases:
            with patch.dict(os.environ, {
                'OPENAI_API_KEY': 'test-key',
                'BATCH_SIZE': batch_size,
                'INSERT_BATCH_SIZE': insert_batch_size
            }):
                loader = FinanceTermLoader()
                assert loader.batch_size == expected_batch


class TestDataProcessing:
    """数据处理相关测试"""
    
    @pytest.fixture
    def mock_loader(self):
        """创建模拟的加载器"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            return FinanceTermLoader()
    
    def test_data_insertion_mock(self, mock_loader):
        """测试数据插入（模拟）"""
        # 创建测试数据
        df = pd.DataFrame({
            'term': ['银行', '贷款'],
            'category': ['FINTERM', 'FINTERM']
        })
        embeddings = [[0.1] * 3072, [0.2] * 3072]
        
        # 模拟Milvus集合
        mock_collection = MagicMock()
        
        # 模拟插入批次大小
        with patch.dict(os.environ, {'INSERT_BATCH_SIZE': '1'}):
            result = mock_loader.insert_data(mock_collection, df, embeddings)
            
            # 验证插入调用
            assert mock_collection.insert.call_count == 2  # 两批次
            assert mock_collection.flush.called
            assert result == 2  # 返回插入的记录数
    
    def test_embedding_batch_processing(self, mock_loader):
        """测试嵌入向量批处理"""
        texts = ['银行', '贷款', '投资', '股票']  # 4个文本
        
        # 模拟批次大小为2
        with patch.dict(os.environ, {'BATCH_SIZE': '2'}):
            loader = FinanceTermLoader()
            
            # 模拟嵌入模型
            mock_embeddings = MagicMock()
            mock_embeddings.embed_documents.side_effect = [
                [[0.1] * 3072, [0.2] * 3072],  # 第一批
                [[0.3] * 3072, [0.4] * 3072]   # 第二批
            ]
            
            with patch.object(loader, 'embeddings', mock_embeddings):
                embeddings = loader.generate_embeddings(texts)
                
                assert len(embeddings) == 4
                assert mock_embeddings.embed_documents.call_count == 2
                
                # 验证批次调用
                calls = mock_embeddings.embed_documents.call_args_list
                assert calls[0][0][0] == ['银行', '贷款']  # 第一批
                assert calls[1][0][0] == ['投资', '股票']  # 第二批


class TestErrorHandling:
    """错误处理测试"""
    
    def test_invalid_csv_structure(self):
        """测试无效的CSV文件结构"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            loader = FinanceTermLoader()
            
            # 创建结构错误的CSV
            invalid_data = pd.DataFrame({
                'wrong_column': ['data1', 'data2'],
                'another_wrong': ['data3', 'data4']
            })
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                invalid_data.to_csv(f.name, index=False)
                csv_path = f.name
            
            try:
                # 应该处理错误或给出有意义的错误信息
                with pytest.raises(Exception):
                    loader.read_csv_data(csv_path)
            finally:
                os.unlink(csv_path)
    
    def test_empty_csv_file(self):
        """测试空CSV文件"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            loader = FinanceTermLoader()
            
            # 创建空CSV文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                f.write('term,category\n')  # 只有header
                csv_path = f.name
            
            try:
                df = loader.read_csv_data(csv_path)
                assert len(df) == 0
            finally:
                os.unlink(csv_path)


class TestIntegration:
    """集成测试"""
    
    @patch('src.finance_term_loader.connections')
    @patch('src.finance_term_loader.Collection')
    @patch('src.finance_term_loader.utility')
    def test_full_workflow_mock(self, mock_utility, mock_collection_class, mock_connections):
        """测试完整工作流程（使用模拟）"""
        # 模拟环境
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test-key',
            'TEST_MODE_LIMIT': '2'
        }):
            loader = FinanceTermLoader()
            
            # 模拟嵌入模型
            mock_embeddings = MagicMock()
            mock_embeddings.embed_documents.return_value = [
                [0.1] * 3072,
                [0.2] * 3072
            ]
            
            # 模拟集合
            mock_collection = MagicMock()
            mock_collection_class.return_value = mock_collection
            mock_utility.has_collection.return_value = False
            
            # 创建测试CSV
            test_data = pd.DataFrame({
                'term': ['银行', '贷款', '投资'],
                'category': ['FINTERM', 'FINTERM', 'FINTERM']
            })
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
                test_data.to_csv(f.name, index=False, encoding='utf-8')
                csv_path = f.name
            
            try:
                with patch.object(loader, 'embeddings', mock_embeddings):
                    result = loader.load_finance_terms(csv_path)
                    
                    # 验证结果
                    assert result['total_terms'] == 2  # 测试模式限制
                    assert result['collection_name'] == 'finance_term'
                    assert result['embedding_model'] == 'text-embedding-3-large'
                    
                    # 验证调用链
                    mock_connections.connect.assert_called()
                    mock_collection_class.assert_called()
                    mock_collection.insert.assert_called()
                    mock_collection.load.assert_called()
                    
            finally:
                os.unlink(csv_path) 