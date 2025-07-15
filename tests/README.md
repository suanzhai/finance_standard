# 金融术语加载工具 - 测试套件

这个目录包含了金融术语加载工具的完整测试套件，将原本的使用示例转换为结构化的单元测试。

## 测试结构

### 📁 测试文件

- `test_configuration.py` - 配置管理测试
  - 环境变量加载
  - 参数验证  
  - 配置文件处理
  - 默认值测试

- `test_finance_term_loader.py` - 核心功能测试
  - CSV文件读取
  - 嵌入向量生成
  - Milvus连接和集合创建
  - 数据插入
  - 完整工作流程

- `test_batch_processing.py` - 批处理功能测试
  - 批次大小配置
  - 嵌入向量批处理
  - 数据插入批处理
  - 进度追踪
  - 边界情况测试

## 🚀 运行测试

### 方法一：使用测试脚本（推荐）

```bash
# 安装测试依赖
python tests/run_tests.py --install

# 运行所有快速测试（不需要API密钥）
python tests/run_tests.py

# 运行特定测试
python tests/run_tests.py --test config    # 配置测试
python tests/run_tests.py --test loader    # 核心功能测试  
python tests/run_tests.py --test batch     # 批处理测试
```

### 方法二：直接使用pytest

```bash
# 安装依赖
pip install pytest pytest-mock

# 运行所有测试
pytest tests/ -v

# 运行特定文件
pytest tests/test_configuration.py -v
pytest tests/test_finance_term_loader.py -v
pytest tests/test_batch_processing.py -v

# 只运行快速测试（不需要API）
pytest tests/ -v -m "not slow"
```

## 🧪 测试类型

### 快速测试（Mock）
- 使用模拟对象，不需要真实的API密钥
- 测试核心逻辑和配置处理
- 运行速度快，适合开发阶段

### 集成测试（需要API）
- 需要真实的OpenAI API密钥
- 测试完整的工作流程
- 标记为 `@pytest.mark.slow`

## 📊 测试覆盖

### 配置管理测试
- ✅ 环境变量加载和验证
- ✅ 参数覆盖机制
- ✅ 默认值处理
- ✅ 错误处理（缺失API密钥等）
- ✅ 配置文件加载
- ✅ 无效配置处理

### 核心功能测试  
- ✅ CSV文件读取和解析
- ✅ 测试模式限制
- ✅ 嵌入向量生成（模拟）
- ✅ Milvus连接（Lite和服务器模式）
- ✅ 集合创建和管理
- ✅ 数据插入
- ✅ 完整工作流程

### 批处理功能测试
- ✅ 批次大小配置
- ✅ 嵌入向量批处理逻辑
- ✅ 数据库插入批处理
- ✅ 进度追踪和时间估算
- ✅ 边界情况（单项、大批次等）
- ✅ 不同配置组合

### 错误处理测试
- ✅ 文件不存在
- ✅ 无效CSV结构
- ✅ 空数据处理
- ✅ 网络连接失败
- ✅ 配置错误

## 🔧 测试配置

### pytest.ini
测试配置文件，定义了：
- 测试发现规则
- 标记定义（unit, integration, slow, fast）
- 输出格式
- 警告处理

### 环境变量
测试中使用的环境变量：
- `OPENAI_API_KEY` - OpenAI API密钥（集成测试需要）
- `BATCH_SIZE` - 批处理大小
- `INSERT_BATCH_SIZE` - 插入批次大小
- `TEST_MODE_LIMIT` - 测试模式限制

## 🎯 测试策略

### Mock vs Real
- **Mock测试**：模拟外部依赖，专注于逻辑测试
- **集成测试**：使用真实API和数据库，验证端到端功能

### 数据隔离
- 每个测试使用独立的环境变量
- 临时文件自动清理
- 模拟对象避免副作用

### 参数化测试
- 使用 `pytest.mark.parametrize` 测试多种配置
- 覆盖各种边界情况和配置组合

## 📈 添加新测试

### 1. 测试文件命名
- 文件名：`test_<模块名>.py`
- 类名：`Test<功能名>`
- 方法名：`test_<具体功能>`

### 2. 使用Fixtures
```python
@pytest.fixture
def mock_loader():
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
        return FinanceTermLoader()
```

### 3. 使用Mock
```python
with patch.object(loader, 'embeddings', mock_embeddings):
    result = loader.generate_embeddings(texts)
```

### 4. 参数化测试
```python
@pytest.mark.parametrize("input,expected", [
    ('100', 100),
    ('500', 500),
    ('', 500),  # 默认值
])
def test_batch_size_config(input, expected):
    # 测试代码
```

## 🐛 调试测试

### 详细输出
```bash
pytest tests/ -v -s  # -s 显示print输出
```

### 单个测试
```bash
pytest tests/test_configuration.py::TestConfiguration::test_environment_variable_loading -v
```

### 调试模式
```bash
pytest tests/ --pdb  # 测试失败时进入调试器
```

## 📝 测试报告

测试运行后会显示：
- ✅ 通过的测试数量
- ❌ 失败的测试详情
- ⚠️ 跳过的测试
- 📊 测试覆盖率（如果安装了coverage插件）

## 🔍 持续改进

### 添加覆盖率报告
```bash
pip install pytest-cov
pytest tests/ --cov=src --cov-report=html
```

### 并行测试
```bash  
pip install pytest-xdist
pytest tests/ -n auto  # 自动并行
```

### 性能测试
```bash
pip install pytest-benchmark
# 在测试中使用 benchmark fixture
``` 