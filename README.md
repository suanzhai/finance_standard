# 金融术语加载工具

这个工具用于将CSV文件中的金融术语加载到Milvus向量数据库中，使用LangChain框架和OpenAI的text-embedding-3-large模型生成嵌入向量。

## 功能特性

- 读取CSV格式的金融术语数据
- 基于LangChain框架，使用OpenAI text-embedding-3-large模型生成高质量嵌入向量
- **🔧 自动配置加载** - 智能搜索并加载 `.env` 或 `config.env` 配置文件
- **⚡ 性能优化** - 大批处理、减少延迟、智能进度显示和时间估算
- **🧪 测试模式** - 支持先处理少量数据验证配置，避免长时间等待
- **💾 Milvus Lite** - 默认使用本地文件数据库，无需启动服务
- 自动创建Milvus集合和索引
- 完整的错误处理和日志记录
- 支持大规模数据处理（15000+条术语）
- 易于扩展到其他嵌入模型提供商

## 安装依赖

```bash
pip install -r requirements.txt
```

## 环境准备

### 1. Milvus数据库设置

#### 方式一：使用Milvus Lite（推荐，无需额外设置）

默认配置已经设置为使用Milvus Lite，这是一个本地文件数据库，无需启动任何服务：

```bash
# 无需任何设置，直接运行程序即可
# Milvus Lite会自动在 db/ 目录下创建数据库文件
python src/finance_term_loader.py
```

#### 方式二：使用Milvus服务器（需要Docker或手动安装）

如果你需要使用完整的Milvus服务器，可以：

1. **修改配置文件**：
```bash
# 编辑 config.env 文件
MILVUS_USE_LITE=false
```

2. **启动Milvus服务**：
```bash
# 使用Docker启动Milvus
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker-compose up -d
```

### 2. 配置环境变量

#### 方法一：使用配置文件（推荐，自动加载）

```bash
# 步骤1：编辑配置文件，填入你的API密钥
vim config.env

# 步骤2：直接运行程序（会自动加载配置）
python src/finance_term_loader.py
```

**配置文件说明：**
- 程序会自动搜索并加载 `.env` 或 `config.env` 文件
- 配置文件位置：项目根目录或 `src/` 目录
- 只需要修改 `OPENAI_API_KEY=your-actual-api-key`
- 其他配置使用默认值即可

#### 方法二：直接设置环境变量

```bash
# 必需配置
export OPENAI_API_KEY='your-openai-api-key-here'

# 可选配置（有默认值）
export MILVUS_HOST='localhost'
export MILVUS_PORT='19530'
export MILVUS_COLLECTION_NAME='finance_term'
export BATCH_SIZE='100'

# 运行程序
python src/finance_term_loader.py
```

## 使用方法

### 🚀 快速开始（推荐）

#### 方式一：快速测试（强烈推荐）

```bash
# 1. 运行配置向导（如果还没配置）
python setup_config.py

# 2. 快速测试（运行测试套件验证配置）
python run_tests.py

# 3. 测试成功后可选择继续处理全部数据
```

#### 方式二：直接运行完整程序

```bash
# 运行配置向导，按提示输入API密钥
python setup_config.py

# 配置完成后，直接运行程序（处理全部15000+条数据）
python src/finance_term_loader.py
```

#### 方式三：手动编辑配置文件

1. **配置API密钥**：编辑 `config.env` 文件中的 `OPENAI_API_KEY`
2. **直接运行**：程序会自动加载配置并运行

```bash
# 编辑配置（只需修改API密钥）
vim config.env

# 运行主程序（自动加载配置）
python src/finance_term_loader.py

# 运行测试验证功能（推荐）
python run_tests.py
```

### 其他使用方式

```bash
# 运行测试套件
python run_tests.py

# 运行特定测试
python run_tests.py --test config
python run_tests.py --test loader  
python run_tests.py --test batch
```

### 直接使用类

```python
from finance_term_loader import FinanceTermLoader

# 方法1：使用环境变量配置（推荐）
loader = FinanceTermLoader()

# 方法2：通过参数传入配置
loader = FinanceTermLoader(
    openai_api_key="your-api-key",
    milvus_host="localhost",
    milvus_port=19530
)

# 加载金融术语
result = loader.load_finance_terms("data/万条金融标准术语.csv")
print(f"加载完成，共处理{result['total_terms']}条术语")
```

## 数据结构

### CSV文件格式
CSV文件应包含两列：
- 第一列：术语名称
- 第二列：类别标识（如'FINTERM'）

### Milvus集合结构
- `id`: 自动生成的主键
- `term`: 金融术语名称（VARCHAR，最大500字符）
- `category`: 术语类别（VARCHAR，最大100字符）
- `embedding`: 嵌入向量（3072维FLOAT_VECTOR）

## 配置参数

### 🔧 自动配置加载

程序启动时会按以下顺序自动搜索配置文件：
1. 项目根目录的 `.env` 文件
2. 项目根目录的 `config.env` 文件  
3. `src/` 目录下的 `.env` 文件
4. `src/` 目录下的 `config.env` 文件

找到配置文件后会自动加载，如果都没找到则使用系统环境变量。

### 环境变量配置

| 环境变量 | 默认值 | 说明 | 是否必需 |
|----------|--------|------|----------|
| `OPENAI_API_KEY` | 无 | OpenAI API密钥 | ✅ 必需 |
| `MILVUS_HOST` | `localhost` | Milvus服务器地址 | 可选 |
| `MILVUS_PORT` | `19530` | Milvus服务器端口 | 可选 |
| `MILVUS_COLLECTION_NAME` | `finance_term` | Milvus集合名称 | 可选 |
| `DB_FILE` | `db/milvus_lite.db` | 数据库文件路径 | 可选 |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI嵌入模型 | 可选 |
| `EMBEDDING_DIM` | `3072` | 嵌入向量维度 | 可选 |
| `INSERT_BATCH_SIZE` | `1000` | 数据库插入批次大小 | 可选 |
| `BATCH_SIZE` | `500` | 批处理大小 | 可选 |
| `TEST_MODE_LIMIT` | `0` | 测试模式限制条数（0=全部） | 可选 |
| `MILVUS_USE_LITE` | `true` | 是否使用Milvus Lite | 可选 |

### 构造函数参数

也可以通过构造函数参数覆盖环境变量配置：

| 参数 | 类型 | 说明 |
|------|------|------|
| `openai_api_key` | `Optional[str]` | OpenAI API密钥 |
| `milvus_host` | `Optional[str]` | Milvus服务器地址 |
| `milvus_port` | `Optional[int]` | Milvus服务器端口 |

## 性能优化

### ⚡ 处理速度优化
- **大批处理**：默认批大小从100增加到500，减少API调用次数
- **减少延迟**：API调用间隔从100ms减少到50ms
- **智能进度显示**：实时显示处理进度、剩余时间估算
- **测试模式**：支持先处理少量数据验证配置

### 📊 时间估算
处理15000+条术语的预估时间：
- **测试模式（100条）**：约1-2分钟
- **完整数据（15885条）**：约1-2小时（取决于网络和API限制）

### 🔧 优化建议
- 首次使用建议运行 `python run_tests.py` 验证配置
- 如需加快速度，可增加 `BATCH_SIZE`（最大2048）
- 网络不稳定时，可减少 `BATCH_SIZE` 避免超时

## 错误处理

工具包含完善的错误处理：
- LangChain内置的API调用失败自动重试机制
- 详细的日志记录
- 异常情况下的资源清理
- 统一的错误处理接口

## LangChain集成优势

### 🚀 多提供商支持
- **OpenAI**: text-embedding-3-large, text-embedding-3-small
- **HuggingFace**: 本地模型，支持离线使用
- **Cohere**: 企业级嵌入模型
- **其他**: 易于扩展到新的提供商

### 🛠️ 统一接口
```python
# 切换提供商只需更改一行代码
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# 或者
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### 📊 性能优化
- 内置重试机制
- 自动批处理
- 缓存支持
- 异步处理支持

### 🔧 扩展性
系统支持扩展到多个嵌入模型提供商：
- 添加新的嵌入模型提供商
- 自定义模型参数
- 比较不同模型性能

## 测试

### 🧪 运行测试

项目包含完整的测试套件，将原本的使用示例转换为结构化的单元测试：

```bash
# 安装测试依赖
python tests/run_tests.py --install

# 运行所有快速测试（使用模拟，不需要API密钥）
python tests/run_tests.py

# 运行特定测试
python tests/run_tests.py --test config    # 配置测试
python tests/run_tests.py --test loader    # 核心功能测试  
python tests/run_tests.py --test batch     # 批处理测试
```

### 📊 测试覆盖

- **配置管理测试** - 环境变量、参数验证、默认值
- **核心功能测试** - CSV读取、嵌入生成、Milvus操作
- **批处理测试** - 批次配置、进度追踪、边界情况
- **错误处理测试** - 各种异常情况的处理

详细测试文档请参考：[tests/README.md](tests/README.md)

## 注意事项

1. 确保有足够的API配额处理大量文本（OpenAI/Cohere）
2. 首次运行会创建集合，如果集合已存在会被删除重建
3. 生成15000+条术语的嵌入向量可能需要较长时间
4. 请确保Milvus服务正常运行
5. 使用HuggingFace模型时，首次运行会下载模型文件 