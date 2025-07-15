import pandas as pd
from langchain_openai import OpenAIEmbeddings
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import os
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import logging

class FinanceTermLoader:
    """
    金融术语加载工具类
    用于读取CSV文件中的金融术语，生成嵌入向量并存储到Milvus数据库
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, milvus_host: Optional[str] = None, milvus_port: Optional[int] = None):
        """
        初始化加载器
        
        Args:
            openai_api_key: OpenAI API密钥，如果不提供则从环境变量OPENAI_API_KEY读取
            milvus_host: Milvus服务器地址，如果不提供则从环境变量MILVUS_HOST读取，默认为localhost
            milvus_port: Milvus服务器端口，如果不提供则从环境变量MILVUS_PORT读取，默认为19530
        """
        # 从环境变量获取配置
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.milvus_host = milvus_host or os.getenv('MILVUS_HOST', 'localhost')
        # 处理端口号，支持无效值
        try:
            self.milvus_port = milvus_port or int(os.getenv('MILVUS_PORT', '19530'))
        except ValueError:
            self.milvus_port = 19530  # 使用默认端口
        
        # 验证必需的配置
        if not self.openai_api_key:
            raise ValueError("OpenAI API密钥未提供。请通过参数传入或设置环境变量OPENAI_API_KEY")
        
        # 设置环境变量给LangChain使用
        os.environ['OPENAI_API_KEY'] = self.openai_api_key
        
        # 其他配置从环境变量获取，提供默认值
        self.collection_name = os.getenv('MILVUS_COLLECTION_NAME', 'finance_term')
        self.db_path = os.getenv('DB_FILE', 'db/milvus_lite.db')
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large')
        self.embedding_dim = int(os.getenv('EMBEDDING_DIM', '3072'))  # text-embedding-3-large的向量维度
        # 处理批次大小，支持空字符串
        batch_size_str = os.getenv('BATCH_SIZE', '500').strip()
        self.batch_size = int(batch_size_str) if batch_size_str else 500  # 批处理大小
        # 处理测试模式限制，支持空字符串
        test_limit_str = os.getenv('TEST_MODE_LIMIT', '0').strip()
        self.test_mode_limit = int(test_limit_str) if test_limit_str else 0  # 测试模式限制
        
        # 初始化LangChain OpenAI嵌入模型
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model
        )
        
        # 配置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def read_csv_data(self, csv_path: str) -> pd.DataFrame:
        """
        读取CSV文件
        
        Args:
            csv_path: CSV文件路径
            
        Returns:
            包含术语数据的DataFrame
        """
        try:
            # 读取CSV文件，假设第一列是术语名称，第二列是类别
            df = pd.read_csv(csv_path, header=None, names=['term', 'category'])
            
            # 如果设置了测试模式限制，只取前N条数据
            if self.test_mode_limit > 0:
                original_count = len(df)
                df = df.head(self.test_mode_limit)
                self.logger.info(f"测试模式：从{original_count}条记录中取前{len(df)}条进行处理")
            else:
                self.logger.info(f"成功读取CSV文件，共{len(df)}条记录")
            
            return df
        except Exception as e:
            self.logger.error(f"读取CSV文件失败: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """
        生成文本嵌入向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小，如果不提供则使用配置中的默认值
            
        Returns:
            嵌入向量列表
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        self.logger.info(f"开始生成嵌入向量，总数: {len(texts)}，批大小: {batch_size}，批次数: {total_batches}")
        
        start_time = time.time()
        
        for i in tqdm(range(0, len(texts), batch_size), 
                     desc="生成嵌入向量", 
                     unit="batch",
                     total=total_batches):
            batch_texts = texts[i:i + batch_size]
            batch_start_time = time.time()
            
            try:
                # 使用LangChain的embed_documents方法
                batch_embeddings = self.embeddings.embed_documents(batch_texts)
                embeddings.extend(batch_embeddings)
                
                batch_time = time.time() - batch_start_time
                processed = i + len(batch_texts)
                
                # 估算剩余时间
                if processed > 0:
                    elapsed_time = time.time() - start_time
                    avg_time_per_item = elapsed_time / processed
                    remaining_items = len(texts) - processed
                    estimated_remaining = avg_time_per_item * remaining_items
                    
                    self.logger.info(
                        f"批次完成: {len(batch_texts)}条，"
                        f"累计处理: {processed}/{len(texts)}，"
                        f"批次耗时: {batch_time:.1f}s，"
                        f"估计剩余: {estimated_remaining/60:.1f}分钟"
                    )
                
                # 减少延迟，只在处理大量数据时适当休息
                if batch_size >= 100:
                    time.sleep(0.05)  # 减少延迟
                    
            except Exception as e:
                self.logger.error(f"生成嵌入向量失败: {e}")
                raise
        
        total_time = time.time() - start_time
        self.logger.info(f"嵌入向量生成完成，总耗时: {total_time/60:.1f}分钟，平均每条: {total_time/len(texts):.2f}秒")
        
        return embeddings
    
    def connect_milvus(self):
        """连接到Milvus数据库"""
        try:
            # 确保db目录存在
            os.makedirs("db", exist_ok=True)
            
            # 检查是否使用本地文件数据库
            use_lite = os.getenv('MILVUS_USE_LITE', 'true').lower() == 'true'
            
            if use_lite:
                # 使用Milvus Lite（本地文件数据库）
                db_file = os.path.join("db", "milvus_lite.db")
                connections.connect(
                    alias="default",
                    uri=db_file
                )
                self.logger.info(f"成功连接到Milvus Lite数据库: {db_file}")
            else:
                # 连接到Milvus服务器
                connections.connect(
                    alias="default",
                    host=self.milvus_host,
                    port=self.milvus_port
                )
                self.logger.info(f"成功连接到Milvus服务器: {self.milvus_host}:{self.milvus_port}")
                
        except Exception as e:
            self.logger.error(f"连接Milvus失败: {e}")
            # 如果连接服务器失败，尝试使用Milvus Lite
            if not os.getenv('MILVUS_USE_LITE', 'true').lower() == 'true':
                self.logger.info("尝试使用Milvus Lite作为备选方案...")
                try:
                    db_file = os.path.join("db", "milvus_lite.db")
                    connections.connect(
                        alias="default",
                        uri=db_file
                    )
                    self.logger.info(f"成功连接到Milvus Lite数据库: {db_file}")
                except Exception as lite_error:
                    self.logger.error(f"Milvus Lite连接也失败: {lite_error}")
                    raise
            else:
                raise
    
    def create_collection(self):
        """创建Milvus集合"""
        try:
            # 如果集合已存在，删除它
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                self.logger.info(f"删除已存在的集合: {self.collection_name}")
            
            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="term", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
            ]
            
            # 创建集合schema
            schema = CollectionSchema(fields, description="金融术语集合")
            
            # 创建集合
            collection = Collection(self.collection_name, schema)
            self.logger.info(f"成功创建集合: {self.collection_name}")
            
            return collection
        except Exception as e:
            self.logger.error(f"创建集合失败: {e}")
            raise
    
    def create_index(self, collection: Collection):
        """为向量字段创建索引"""
        try:
            index_params = {
                "metric_type": "COSINE",
                "index_type": "AUTOINDEX",
                "params": {"nlist": 128}
            }
            
            collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            self.logger.info("成功创建向量索引")
        except Exception as e:
            self.logger.error(f"创建索引失败: {e}")
            raise
    
    def insert_data(self, collection: Collection, df: pd.DataFrame, embeddings: List[List[float]]):
        """分批插入数据到集合"""
        try:
            # 获取配置的批次大小，用于插入，支持空字符串
            insert_batch_size_str = os.getenv('INSERT_BATCH_SIZE', '1000').strip()
            insert_batch_size = int(insert_batch_size_str) if insert_batch_size_str else 1000
            total_records = len(df)
            total_batches = (total_records + insert_batch_size - 1) // insert_batch_size
            
            self.logger.info(f"开始分批插入数据，总计 {total_records} 条，分 {total_batches} 批，每批 {insert_batch_size} 条")
            
            inserted_count = 0
            
            # 分批插入数据
            for batch_idx in tqdm(range(total_batches), desc="插入数据", unit="batch"):
                start_idx = batch_idx * insert_batch_size
                end_idx = min(start_idx + insert_batch_size, total_records)
                
                # 准备当前批次的数据
                batch_data = [
                    df['term'].iloc[start_idx:end_idx].tolist(),
                    df['category'].iloc[start_idx:end_idx].tolist(),
                    embeddings[start_idx:end_idx]
                ]
                
                # 插入当前批次
                result = collection.insert(batch_data)
                current_batch_size = end_idx - start_idx
                inserted_count += current_batch_size
                
                self.logger.info(f"批次 {batch_idx + 1}/{total_batches} 插入完成: {current_batch_size} 条，累计: {inserted_count}/{total_records}")
                
                # 如果是大批次，中间刷新一下
                if batch_idx % 5 == 0 or batch_idx == total_batches - 1:
                    collection.flush()
                    self.logger.info(f"已刷新数据到磁盘，当前进度: {inserted_count}/{total_records}")
            
            # 最终刷新集合以确保所有数据被写入
            collection.flush()
            
            self.logger.info(f"成功插入 {inserted_count} 条记录到集合 '{self.collection_name}'")
            return inserted_count
            
        except Exception as e:
            self.logger.error(f"插入数据失败: {e}")
            raise
    
    def load_finance_terms(self, csv_path: str):
        """
        主方法：加载金融术语到Milvus数据库
        
        Args:
            csv_path: CSV文件路径
        """
        try:
            # 1. 读取CSV数据
            self.logger.info("开始读取CSV文件...")
            df = self.read_csv_data(csv_path)
            
            # 2. 生成嵌入向量
            self.logger.info("开始生成嵌入向量...")
            embeddings = self.generate_embeddings(df['term'].tolist())
            
            # 3. 连接Milvus
            self.logger.info("连接Milvus数据库...")
            self.connect_milvus()
            
            # 4. 创建集合
            self.logger.info("创建集合...")
            collection = self.create_collection()
            
            # 5. 插入数据
            self.logger.info("插入数据...")
            self.insert_data(collection, df, embeddings)
            
            # 6. 创建索引
            self.logger.info("创建索引...")
            self.create_index(collection)
            
            # 7. 加载集合
            collection.load()
            
            self.logger.info("金融术语加载完成！")
            
            # 返回统计信息
            return {
                "total_terms": len(df),
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
                "embedding_dim": self.embedding_dim
            }
            
        except Exception as e:
            self.logger.error(f"加载金融术语失败: {e}")
            raise

    def search_similar_terms(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        使用milvusClient（pymilvus）进行相似性搜索。
        Args:
            query: 用户输入的查询词
            top_k: 返回最相近的结果数
        Returns:
            包含术语、类别和相似度分数的字典列表
        """
        from pymilvus import MilvusClient
        # 1. 生成查询词的embedding
        embedding = self.embeddings.embed_query(query)
        # 2. 初始化milvusClient
        uri = self.db_path if os.getenv('MILVUS_USE_LITE', 'true').lower() == 'true' else f"{self.milvus_host}:{self.milvus_port}"
        client = MilvusClient(uri=uri)
        # 3. 查询
        search_result = client.search(
            collection_name=self.collection_name,
            data=[embedding],
            limit=top_k,
            output_fields=["term", "category"],
            search_params={"metric_type": "COSINE"}
        )
        # 4. 解析结果
        output = []
        for hit in search_result[0]:
            output.append({
                "term":  hit['entity'].get("term") or "",
                "category":  hit['entity'].get("category") or "",
                "score": hit.get("distance") or hit.get("score") or 0
            })
        return output

    def init_milvus_data(self, csv_path: str = "data/万条金融标准术语.csv"):
        """
        初始化Milvus数据库并加载金融术语数据。
        Args:
            csv_path: 金融术语CSV文件路径，默认为 data/万条金融标准术语.csv
        Returns:
            统计信息字典
        """
        return self.load_finance_terms(csv_path)

    def process_and_import(self, df, on_embed_progress=None, on_insert_progress=None):
        """
        处理DataFrame，分批生成嵌入并分批插入Milvus，支持进度回调。
        Args:
            df: 金融术语DataFrame，需包含'term'和'category'列
            on_embed_progress: 嵌入进度回调，参数(done, total)
            on_insert_progress: 插入进度回调，参数(done, total)
        Returns:
            统计信息字典
        """
        texts = df['term'].tolist()
        batch_size = self.batch_size
        embeddings = []
        total = len(texts)
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            embeddings.extend(batch_embeddings)
            if on_embed_progress:
                on_embed_progress(i + len(batch_texts), total)
        # 连接Milvus
        self.connect_milvus()
        collection = self.create_collection()
        # 插入数据
        insert_batch_size = int(os.getenv('INSERT_BATCH_SIZE', '1000'))
        total_records = len(df)
        inserted_count = 0
        for batch_idx in range(0, total_records, insert_batch_size):
            start_idx = batch_idx
            end_idx = min(start_idx + insert_batch_size, total_records)
            batch_data = [
                df['term'].iloc[start_idx:end_idx].tolist(),
                df['category'].iloc[start_idx:end_idx].tolist(),
                embeddings[start_idx:end_idx]
            ]
            collection.insert(batch_data)
            inserted_count = end_idx
            if on_insert_progress:
                on_insert_progress(inserted_count, total_records)
        collection.flush()
        self.create_index(collection)
        collection.load()
        return {
            "total_terms": len(df),
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim
        }


# 使用示例
if __name__ == "__main__":
    # 使用示例
    # 方法一：通过环境变量配置（推荐）
    # export OPENAI_API_KEY="your-openai-api-key-here"
    # export MILVUS_HOST="localhost"
    # export MILVUS_PORT="19530"
    # export MILVUS_COLLECTION_NAME="finance-term"
    
    try:
        # 创建加载器实例（使用环境变量配置）
        loader = FinanceTermLoader()
        
        # 加载金融术语
        result = loader.init_milvus_data()
        
        print(f"加载完成！统计信息: {result}")
        
    except ValueError as e:
        print(f"配置错误: {e}")
        print("请设置必要的环境变量或通过参数传入配置")
    except Exception as e:
        print(f"加载失败: {e}") 